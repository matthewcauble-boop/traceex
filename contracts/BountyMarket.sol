// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

/// @title BountyMarket: free-to-post bounties, each with its own coin on a linear bonding curve (paid in USDC).
/// @notice Same maths as sdk/python/traceex/bountycoin.py.
///   price(s) = BASE + SLOPE * s   (USDC micros per whole coin; coins carry 6 decimals, so 1e6 units = 1 coin)
///   Buying funds the bounty's pool; while open, holders can sell back to the curve. Coins transfer freely.
///   solve():   the exchange's settlement key moves the pool into the epoch payout (solver 70 / traces 20 /
///              checkers 5 / validators 5, paid by PayoutDistributor) once a validator-attested learning meets the
///              bounty's target on its hidden eval set. From then on the coin is a share of that solution.
///   payHolders(): every epoch the settlement key deposits the holders' cut (20%) of the solution's metered revenue.
///              Each coin accrues it; holders withdraw whenever they like. Accrual follows the coin, so a transfer
///              never moves earnings that were accrued before it.
///   expire():  unsolved past the deadline: every holder redeems coins for a pro-rata share of the pool.
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract BountyMarket {
    uint256 public constant BASE = 10_000;        // $0.01 per coin at zero supply
    uint256 public constant SLOPE = 100;          // +$0.0001 per coin of supply
    uint256 constant ONE = 1e6;                   // coin decimals
    uint256 constant ACC = 1e18;                  // accrual precision

    enum Status { Open, Solved, Expired }
    struct Bounty {
        address poster; bytes32 pathHash; bytes32 evalSet; uint32 targetBps; uint64 deadline;
        Status status; uint256 supply; uint256 pool; bytes32 learning; uint256 accPerCoin;
    }

    IERC20 public immutable usdc;
    address public settler;                       // the exchange's settlement key (multisig in production)
    Bounty[] public bounties;
    mapping(uint256 => mapping(address => uint256)) public balanceOf;
    mapping(uint256 => mapping(address => uint256)) internal _paidAcc;   // accPerCoin already counted for the holder
    mapping(uint256 => mapping(address => uint256)) public owed;         // accrued, not yet withdrawn

    event Posted(uint256 indexed id, address indexed poster, bytes32 pathHash, bytes32 evalSet, uint32 targetBps, uint64 deadline);
    event Bought(uint256 indexed id, address indexed buyer, uint256 coins, uint256 cost);
    event Sold(uint256 indexed id, address indexed seller, uint256 coins, uint256 paid);
    event Transfer(uint256 indexed id, address indexed from, address indexed to, uint256 coins);
    event Solved(uint256 indexed id, bytes32 learning, uint256 pool);
    event HoldersPaid(uint256 indexed id, uint256 amount);
    event Expired(uint256 indexed id, uint256 pool);

    constructor(IERC20 _usdc, address _settler) { usdc = _usdc; settler = _settler; }

    // ---- curve ---------------------------------------------------------------------------------------------------
    /// USDC micros to buy `n` coin units starting at supply `s` (both in 1e6 units). Rounds up for buyers.
    function cost(uint256 s, uint256 n) public pure returns (uint256) { return _area(s, n, true); }

    /// USDC micros paid for selling `n` units back from supply `s + n`. Rounds down for sellers, so the pool (sum of
    /// rounded-up buys) always covers the exact area under the curve and no seller can eat into another's share.
    function sellValue(uint256 s, uint256 n) public pure returns (uint256) { return _area(s, n, false); }

    /// Exact area under price(x) = BASE + SLOPE * x over [s, s + n], as one fraction, rounded once at the end:
    ///   (2 * ONE * BASE * n + SLOPE * (2 * s * n + n * n)) / (2 * ONE * ONE)
    function _area(uint256 s, uint256 n, bool up) internal pure returns (uint256) {
        uint256 num = 2 * ONE * BASE * n + SLOPE * (2 * s * n + n * n);
        uint256 den = 2 * ONE * ONE;
        return up ? (num + den - 1) / den : num / den;
    }
    function priceNow(uint256 id) external view returns (uint256) { return BASE + SLOPE * bounties[id].supply / ONE; }

    // ---- lifecycle -----------------------------------------------------------------------------------------------
    function post(bytes32 pathHash, bytes32 evalSet, uint32 targetBps, uint64 epochsOpen) external returns (uint256 id) {
        id = bounties.length;
        bounties.push(Bounty(msg.sender, pathHash, evalSet, targetBps, uint64(block.timestamp) + epochsOpen * 1 days,
                             Status.Open, 0, 0, bytes32(0), 0));
        emit Posted(id, msg.sender, pathHash, evalSet, targetBps, uint64(block.timestamp) + epochsOpen * 1 days);
    }

    function buy(uint256 id, uint256 coins, uint256 maxCost) external {
        Bounty storage b = bounties[id];
        require(b.status == Status.Open && block.timestamp < b.deadline, "not open");
        uint256 c = cost(b.supply, coins);
        require(c <= maxCost, "slippage");
        require(usdc.transferFrom(msg.sender, address(this), c), "pay failed");
        _accrue(id, msg.sender);
        b.supply += coins; b.pool += c; balanceOf[id][msg.sender] += coins;
        emit Bought(id, msg.sender, coins, c);
    }

    function sell(uint256 id, uint256 coins, uint256 minOut) external {
        Bounty storage b = bounties[id];
        require(b.status == Status.Open && block.timestamp < b.deadline, "not open");
        require(balanceOf[id][msg.sender] >= coins, "balance");
        uint256 v = sellValue(b.supply - coins, coins);
        if (v > b.pool) v = b.pool;                           // belt and braces: never pay more than the pool
        require(v >= minOut, "slippage");
        b.supply -= coins; b.pool -= v; balanceOf[id][msg.sender] -= coins;
        require(usdc.transfer(msg.sender, v), "pay failed");
        emit Sold(id, msg.sender, coins, v);
    }

    function transfer(uint256 id, address to, uint256 coins) external {
        require(to != address(0), "zero address");           // burnt coins would keep diluting accrual and redeem
        require(balanceOf[id][msg.sender] >= coins, "balance");
        _accrue(id, msg.sender); _accrue(id, to);
        balanceOf[id][msg.sender] -= coins; balanceOf[id][to] += coins;
        emit Transfer(id, msg.sender, to, coins);
    }

    /// The settlement key solves a bounty after verifying the learning's attestation off-chain (Registry holds the
    /// attestation hash). The pool goes to the settler, who pays it out in the epoch's Merkle root.
    function solve(uint256 id, bytes32 learning) external {
        Bounty storage b = bounties[id];
        require(msg.sender == settler && b.status == Status.Open && block.timestamp < b.deadline, "cannot solve");   // a deadline is a deadline
        b.status = Status.Solved; b.learning = learning;
        uint256 p = b.pool; b.pool = 0;
        require(usdc.transfer(settler, p), "pay failed");
        emit Solved(id, learning, p);
    }

    /// Each epoch: deposit the holders' cut of the solution's revenue; every coin accrues its share.
    function payHolders(uint256 id, uint256 amount) external {
        Bounty storage b = bounties[id];
        require(msg.sender == settler && b.status == Status.Solved && b.supply > 0, "cannot pay");
        require(usdc.transferFrom(msg.sender, address(this), amount), "fund failed");
        b.accPerCoin += amount * ACC / b.supply;
        emit HoldersPaid(id, amount);
    }

    function withdraw(uint256 id) external returns (uint256 amt) {
        _accrue(id, msg.sender);
        amt = owed[id][msg.sender]; owed[id][msg.sender] = 0;
        require(usdc.transfer(msg.sender, amt), "pay failed");
    }

    function expire(uint256 id) external {
        Bounty storage b = bounties[id];
        require(b.status == Status.Open && block.timestamp >= b.deadline, "not expirable");
        b.status = Status.Expired;
        emit Expired(id, b.pool);
    }

    /// After expiry, burn coins for a pro-rata share of what is left of the pool.
    function redeem(uint256 id) external returns (uint256 amt) {
        Bounty storage b = bounties[id];
        require(b.status == Status.Expired, "not expired");
        uint256 bal = balanceOf[id][msg.sender];
        require(bal > 0, "nothing to redeem");
        amt = b.pool * bal / b.supply;
        b.pool -= amt; b.supply -= bal; balanceOf[id][msg.sender] = 0;
        require(usdc.transfer(msg.sender, amt), "pay failed");
    }

    function pending(uint256 id, address who) external view returns (uint256) {
        return owed[id][who] + balanceOf[id][who] * (bounties[id].accPerCoin - _paidAcc[id][who]) / ACC;
    }

    function _accrue(uint256 id, address who) internal {
        uint256 acc = bounties[id].accPerCoin;
        owed[id][who] += balanceOf[id][who] * (acc - _paidAcc[id][who]) / ACC;
        _paidAcc[id][who] = acc;
    }

    function count() external view returns (uint256) { return bounties.length; }
}
