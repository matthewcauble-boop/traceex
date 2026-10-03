// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

/// @title PayoutDistributor: one Merkle root per epoch, everyone claims their own USDC with a proof.
/// @notice Leaf = sha256(abi.encodePacked(uint64 epoch, address account, uint256 amount)); pairs hashed sorted.
///         Matches sdk/python/traceex/merkle.py byte for byte. One root write per epoch keeps per-payee cost at a
///         single claim, which a payee can batch across many epochs.
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract PayoutDistributor {
    IERC20 public immutable usdc;
    address public poster;                       // the exchange's settlement key (a multisig in production)
    uint64 public constant CHALLENGE = 1 days;   // anyone can recompute the root from the public epoch log

    struct Epoch { bytes32 root; uint256 total; uint64 postedAt; bool vetoed; uint256 paid; }
    mapping(uint64 => Epoch) public epochs;
    mapping(uint64 => mapping(address => bool)) public claimed;

    event RootPosted(uint64 indexed epoch, bytes32 root, uint256 total);
    event Vetoed(uint64 indexed epoch);
    event Claimed(uint64 indexed epoch, address indexed account, uint256 amount);

    constructor(IERC20 _usdc, address _poster) { usdc = _usdc; poster = _poster; }

    /// The poster funds the epoch in the same call, so a root can never promise more than it holds.
    function postRoot(uint64 epoch, bytes32 root, uint256 total) external {
        require(msg.sender == poster, "not poster");
        require(root != bytes32(0), "zero root");   // a zero root reads as "unposted": funds would be stranded
        require(epochs[epoch].root == bytes32(0) || epochs[epoch].vetoed, "epoch exists");   // a vetoed epoch can be re-posted
        require(usdc.transferFrom(msg.sender, address(this), total), "fund failed");
        epochs[epoch] = Epoch(root, total, uint64(block.timestamp), false, 0);
        emit RootPosted(epoch, root, total);
    }

    /// Inside the challenge window the poster can withdraw a wrong root (e.g. a validator proved a bad attestation).
    function veto(uint64 epoch) external {
        Epoch storage e = epochs[epoch];
        require(msg.sender == poster && e.root != bytes32(0) && !e.vetoed && block.timestamp < e.postedAt + CHALLENGE, "cannot veto");
        e.vetoed = true;
        require(usdc.transfer(poster, e.total), "refund failed");
        emit Vetoed(epoch);
    }

    function claim(uint64 epoch, address account, uint256 amount, bytes32[] calldata proof) public {
        Epoch storage e = epochs[epoch];
        require(e.root != bytes32(0) && !e.vetoed, "no root");
        require(block.timestamp >= e.postedAt + CHALLENGE, "challenge window");
        require(!claimed[epoch][account], "claimed");
        bytes32 h = sha256(abi.encodePacked(epoch, account, amount));
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 s = proof[i];
            h = h < s ? sha256(abi.encodePacked(h, s)) : sha256(abi.encodePacked(s, h));
        }
        require(h == e.root, "bad proof");
        require(e.paid + amount <= e.total, "exceeds epoch");   // a bad root can only spend its own epoch's money
        e.paid += amount;
        claimed[epoch][account] = true;
        require(usdc.transfer(account, amount), "transfer failed");   // paid to the account, whoever submits
        emit Claimed(epoch, account, amount);
    }

    function claimMany(uint64[] calldata epochs_, address account, uint256[] calldata amounts, bytes32[][] calldata proofs) external {
        for (uint256 i = 0; i < epochs_.length; i++) claim(epochs_[i], account, amounts[i], proofs[i]);
    }
}
