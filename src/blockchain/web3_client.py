import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3


BASE_DIR = Path(__file__).resolve().parents[2]
ABI_PATH = BASE_DIR / "src" / "blockchain" / "contract_abi.json"


class BlockchainClient:
    def __init__(self) -> None:
        load_dotenv()

        self.provider_uri = os.getenv("WEB3_PROVIDER_URI", "").strip()
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "").strip()
        self.private_key = os.getenv("PRIVATE_KEY", "").strip()
        self.chain_id = int(os.getenv("CHAIN_ID", "11155111"))

        self.is_configured = all([
            self.provider_uri,
            self.contract_address,
            self.private_key
        ])

        self.w3: Optional[Web3] = None
        self.account = None
        self.contract = None

        if not self.is_configured:
            return

        self.w3 = Web3(Web3.HTTPProvider(self.provider_uri))
        if not self.w3.is_connected():
            raise ConnectionError("Could not connect to the blockchain RPC provider.")

        self.account = Account.from_key(self.private_key)

        with open(ABI_PATH, "r", encoding="utf-8") as f:
            abi = json.load(f)

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=abi
        )

    def _ensure_ready(self) -> None:
        if not self.is_configured or self.w3 is None or self.contract is None or self.account is None:
            raise RuntimeError("Blockchain client is not configured properly.")

    def is_hash_registered(self, file_hash: str) -> bool:
        self._ensure_ready()
        return bool(self.contract.functions.isHashRegistered(file_hash).call())

    def get_file_record(self, file_hash: str) -> Dict[str, Any]:
        self._ensure_ready()
        exists, timestamp, registered_by = self.contract.functions.getFileRecord(file_hash).call()
        return {
            "exists": bool(exists),
            "timestamp": int(timestamp),
            "registered_by": registered_by
        }

    def register_file_hash(self, file_hash: str) -> Dict[str, Any]:
        self._ensure_ready()

        if self.is_hash_registered(file_hash):
            record = self.get_file_record(file_hash)
            return {
                "success": True,
                "already_registered": True,
                "tx_hash": None,
                "record": record
            }

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        gas_estimate = self.contract.functions.registerFileHash(file_hash).estimate_gas({
            "from": self.account.address
        })

        transaction = self.contract.functions.registerFileHash(file_hash).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "chainId": self.chain_id,
            "gas": gas_estimate + 50000,
            "gasPrice": self.w3.eth.gas_price
        })

        signed_txn = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "success": receipt.status == 1,
            "already_registered": False,
            "tx_hash": tx_hash.hex(),
            "record": self.get_file_record(file_hash)
        }