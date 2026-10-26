from tpu_host.vectors import VECTORS_PATH, generate


def test_shared_vectors_file_is_up_to_date():
    assert VECTORS_PATH.read_text(encoding="utf-8") == generate(), (
        "ref_vectors.txt is stale: run `python -m tpu_host.vectors` in host/ and commit it")
