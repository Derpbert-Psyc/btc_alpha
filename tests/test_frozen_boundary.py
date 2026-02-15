"""Gate 0: Frozen boundary integrity — verify frozen files are unmodified."""

import hashlib
import os
import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

FROZEN_MANIFEST = {
    "composition_compiler_v1_5_2.py": "f22251d40361b9264e698bba22b6be7f0f732c8fa27c3ccb7d23fe55bce6429a",
    "strategy_framework_v1_8_0.py": "905ff13c17f1f7175cd0d06027d6f9c750fffb08357898db78a220c17b25cbc2",
    "verify_framework_v1_8_0.py": "b20cdeda5d8b7b6350230eb1b99faf69b5894d5492212c1b9fa0fb3e39e13b56",
    "phase5_bundle_store.py": "209b0b3bffb0dc130f2296b53ef55d618ff29c322fc369afd14e3dc0147a6a0a",
    "phase5_cli.py": "b8b65db9da3ba4003e62fbb85fe2d67b02db4b655cd2abd257e1887f4e0c7b7a",
    "phase5_integrated_runner.py": "96fd0350be035307ed16ead1ad2c00f270b1aa3e7f9d19e89ee04a1cb7492a28",
    "phase5_oms.py": "c4940e4b0745936d236c1355c6908d581198bddc8eb43c0793f0578d34488993",
    "phase5_oms_types.py": "d5c2dd28be32183baeee3caf8d0f088a0fc32658a46a69b1d4ec9c219e95506f",
    "phase5_pod.py": "0379d078f9bd807698926d7a311e0120936f122a18f3b30f5a7e7ddce510b943",
    "phase5_triage.py": "8dcc2203fe214bf63e876d6691d38217f73aeec250e40ae5a4b537fb9f9913af",
    "phase5_triage_types.py": "c698a7792153ef627823dd7c992aecad80675830c83b45c3c0cd82876ab438b8",
    "phase5_type_bridge.py": "730d635ec64868007b91ecf21b2d1fcbf4142cdacc47ba9be71ed924cf101815",
    "btc_alpha_v3_final.py": "45b30944468934e94af3dc04f49d38434c9adc73f0077f7075b09af660fe4c06",
    "btc_alpha_phase2_v4.py": "2a5ed16e8ddc244a0bdef868a99ebe8e4a8dacb3cf247b8c3de34deab51d8294",
    "btc_alpha_phase3.py": "4a4ccbd1f74c41a41548c22fc20f16e8f1dee938d0c829a488b7ce4bdc8ca724",
    "btc_alpha_phase4b_1_7_2.py": "a2869c769f16c5f0b9ae159f6356482072d656c535eeb735a9286dca45d3252b",
    "SYSTEM_LAWS.md": "e817af5812770bc44b5d55d3b1dcb180d0c38d7129912bc6b9164dfe36626cef",
    "BYBIT_OMS_CONTRACT_v1_0_0.md": "920b984550d34362a0450da23ebd618e411c30a5b8871206e18a706694be4f2a",
    "PHASE2_INVARIANTS.md": "2b5ac60e63d38696c800ef3e6f47fc71ad1d064ea1af8f035b1311c633945a09",
    "PHASE4A_INDICATOR_CONTRACT.md": "0db864460311feeec16ab18553f87f79a513efe4afeb9c612f03fcfa31b5cabe",
    "PHASE4B_CONTRACT_LOCKED.md": "03c4628253698403c761e9a4a398b61ee624532452b9b0cad6a0e1194c09bc91",
    "PHASE5_INTEGRATION_SEAMS_v1_2_3.md": "b02857b543e9af536cf733924403c61c1b88bca4a4a38ca91ff00d49606c1b70",
    "PHASE5_INTEGRATION_SEAMS_v1_2_6.md": "056570dcffc445ad005ec48a25b14d8092263cd33ebc19e299cc668983fcc9cd",
    "PHASE5_ROBUSTNESS_CONTRACT_v1_2_0.md": "26c4504442d8a5af4e6e96465bf8512293e9287557858b658e34ea99a4b39ef3",
}


def _sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.parametrize("filename,expected_hash", list(FROZEN_MANIFEST.items()))
def test_frozen_file_integrity(filename, expected_hash):
    filepath = os.path.join(PROJECT_ROOT, filename)
    assert os.path.exists(filepath), f"Frozen file missing: {filename}"
    actual = _sha256(filepath)
    assert actual == expected_hash, (
        f"Frozen file modified: {filename}\n"
        f"  Expected: {expected_hash}\n"
        f"  Actual:   {actual}"
    )
