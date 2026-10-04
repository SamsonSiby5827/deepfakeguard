// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MediaHashRegistry {
    struct FileRecord {
        bool exists;
        uint256 timestamp;
        address registeredBy;
    }

    mapping(string => FileRecord) private records;

    event FileHashRegistered(
        string fileHash,
        uint256 timestamp,
        address registeredBy
    );

    function registerFileHash(string memory fileHash) public {
        require(bytes(fileHash).length > 0, "Hash cannot be empty");
        require(!records[fileHash].exists, "Hash already registered");

        records[fileHash] = FileRecord({
            exists: true,
            timestamp: block.timestamp,
            registeredBy: msg.sender
        });

        emit FileHashRegistered(fileHash, block.timestamp, msg.sender);
    }

    function isHashRegistered(string memory fileHash) public view returns (bool) {
        return records[fileHash].exists;
    }

    function getFileRecord(string memory fileHash)
        public
        view
        returns (bool exists, uint256 timestamp, address registeredBy)
    {
        FileRecord memory record = records[fileHash];
        return (record.exists, record.timestamp, record.registeredBy);
    }
}