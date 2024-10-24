from datasketch import MinHashLSH, MinHash, LeanMinHash
from datasets import load_dataset, load_from_disk
import os
import pickle
from tqdm import tqdm

SIMILARITY_THRESHOLD = 0.9
NUM_PERMS = 128
SHINGLE_SIZE = 4

lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERMS)
dataset = load_dataset("temp", split="train")


# Apply the function to initialize a column

def add_is_duplicate_column(example):
    example["is_duplicate"] = False  # Initialize to False
    return example


dataset = dataset.map(add_is_duplicate_column, num_proc=os.cpu_count())


def _shingle(string, shingle_size=SHINGLE_SIZE):
    shings = {
        string[i: i + shingle_size].encode("utf8")
        for i in range(len(string) - shingle_size + 1)
    }
    return set(shings)


def get_cassandra_lsh():
    lsh = MinHashLSH(
        threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERMS, storage_config={
            'type': 'cassandra',
            'basename': b'base_lsh_cassandra',
            'cassandra': {
                'seeds': ['127.0.0.1', "cassandra"],
                'keyspace': 'lsh_test',
                'replication': {
                    'class': 'SimpleStrategy',
                    'replication_factor': '1',
                },
                'drop_keyspace': False,
                'drop_tables': False,
            }
        }
    )

    return lsh


def hash_and_insert_with_basename(batch, indices):
    lsh = get_cassandra_lsh()

    with lsh.insertion_session() as session:
        for i, row in enumerate(batch["text"]):
            shingles = _shingle(row, shingle_size=SHINGLE_SIZE)
            # shingles = [shing.encode("utf8") for shing in shingles]

            if len(shingles) != 0:
                minhash = MinHash(num_perm=NUM_PERMS)
                for shing in shingles:
                    minhash.update(shing)

                minhash = LeanMinHash(minhash=minhash)
                session.insert(str(indices[i]),
                               minhash, check_duplication=False)

    return batch


dataset = dataset.map(hash_and_insert_with_basename, batched=True, batch_size=3000,
                      num_proc=os.cpu_count(), with_indices=True)


def marked_duplicate(batch, indices):
    lsh = get_cassandra_lsh()
    for i, row in enumerate(batch["text"]):
        try:
            lsh.__contains__(str(indices[i]))
            shingles = _shingle(row, shingle_size=SHINGLE_SIZE)

            if len(shingles) != 0:
                minhash = MinHash(num_perm=NUM_PERMS)
                for shing in shingles:
                    minhash.update(shing)

                    query = lsh.query(minhash=minhash)

                    if len(query) == 0:
                        batch["is_duplicate"][i] = False
                    else:
                        for id in query:
                            if id == indices[i]:
                                batch["is_duplicate"][i] = False
                            else:
                                lsh.remove(id)
        except Exception:
            continue

    return batch


dataset = dataset.map(marked_duplicate, batched=True,
                      with_indices=True, batch_size=3000, num_proc=os.cpu_count())
dataset = dataset.filter(
    lambda x: x["is_duplicate"] == False, num_proc=os.cpu_count())

dataset.save_to_disk("deduplicated_data")
