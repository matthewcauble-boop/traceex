// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

/// @title Registry: who owns which trace and learning, and which learnings descend from which.
/// @notice Only content hashes go on-chain; the objects themselves live with the exchange nodes (and IPFS if you
///         like). Ownership is the address that first registered a hash. Producers can register their own traces
///         directly, so a node can never claim them. Learnings record parents with weights in basis points; that
///         family tree is what royalties follow.
contract Registry {
    struct Entry { address owner; uint64 registeredAt; uint8 kind; }   // kind: 1 trace, 2 learning
    struct Parent { bytes32 id; uint16 weightBps; }

    mapping(bytes32 => Entry) public entries;
    mapping(bytes32 => Parent[]) internal _parents;
    mapping(bytes32 => bytes32) public attestation;   // learning id => hash of the validator attestation

    event TraceRegistered(bytes32 indexed id, address indexed owner, bytes32 indexed lot);
    event LearningRegistered(bytes32 indexed id, address indexed trainer, bytes32 attestation);
    event Transferred(bytes32 indexed id, address indexed from, address indexed to);

    function registerTrace(bytes32 id, bytes32 lot) external {
        require(entries[id].owner == address(0), "exists");
        entries[id] = Entry(msg.sender, uint64(block.timestamp), 1);
        emit TraceRegistered(id, msg.sender, lot);
    }

    /// Batch form: one transaction for many traces keeps registration at a fraction of a cent each on an L2.
    function registerTraces(bytes32[] calldata ids, bytes32 lot) external {
        for (uint256 i = 0; i < ids.length; i++) {
            if (entries[ids[i]].owner != address(0)) continue;
            entries[ids[i]] = Entry(msg.sender, uint64(block.timestamp), 1);
            emit TraceRegistered(ids[i], msg.sender, lot);
        }
    }

    function registerLearning(bytes32 id, Parent[] calldata ps, bytes32 attestationHash) external {
        require(entries[id].owner == address(0), "exists");
        uint256 sum;
        for (uint256 i = 0; i < ps.length; i++) {
            require(entries[ps[i].id].owner != address(0), "unknown parent");
            sum += ps[i].weightBps;
            _parents[id].push(ps[i]);
        }
        require(sum == 10_000, "weights must sum to 100%");
        entries[id] = Entry(msg.sender, uint64(block.timestamp), 2);
        attestation[id] = attestationHash;
        emit LearningRegistered(id, msg.sender, attestationHash);
    }

    /// Ownership is transferable: sell the future royalties of a trace or learning.
    function transfer(bytes32 id, address to) external {
        require(entries[id].owner == msg.sender, "not owner");
        require(to != address(0), "zero address");   // owner 0 means "unregistered": anyone could re-register the id
        entries[id].owner = to;
        emit Transferred(id, msg.sender, to);
    }

    function parents(bytes32 id) external view returns (Parent[] memory) { return _parents[id]; }
}
