from datasketch import MinHashLSH, MinHash, LeanMinHash
from datasets import load_dataset, load_from_disk, Dataset
from regex import W
from tqdm import tqdm
import multiprocessing as mp


class DeduplicateWithCassandra:
    def __init__(self, dataset, similarity_threshold, num_perms, shingle_size, keyspace, replication_factor: str = "1", strategy: str = "SimpleStrategy", port: str = "127.0.0.1"):
        self.dataset = dataset

        self.similarity_threshold = similarity_threshold
        self.num_perms = num_perms
        self.shingle_size = shingle_size

        self.keyspace = keyspace
        self.replication_factor = replication_factor
        self.strategy = strategy
        self.port = port

    def hash_insert_mark_with_basename(self, batch, indices, lsh):
        for i, row in enumerate(batch["text"]):
            shingles = self._shingle(row, shingle_size=self.shingle_size)

            if len(shingles) != 0:
                minhash = MinHash(num_perm=self.num_perms)
                for shing in shingles:
                    minhash.update(shing)

                minhash = LeanMinHash(minhash=minhash)
                query = lsh.query(minhash)

                if len(query) == 0:
                    lsh.insert(str(indices[i]), minhash,
                               check_duplication=False)
                    batch["is_duplicate"][i] = False
                else:
                    batch["is_duplicate"][i] = True

        return batch

    def get_cassandra_lsh(self):
        lsh = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.num_perms, storage_config={
                'type': 'cassandra',
                'basename': b'lshbase',
                'cassandra': {
                    'seeds': ['127.0.0.1', "cassandra"],
                    'keyspace': self.keyspace,
                    'replication': {
                        'class': self.strategy,
                        'replication_factor': str(self.replication_factor),
                    },
                    'drop_keyspace': False,
                    'drop_tables': False,
                }
            }
        )

        return lsh

    def add_is_duplicate_column(self, example):
        example["is_duplicate"] = False  # Initialize to False
        return example

    def run(self, dataset, cpu_count=8, reset=True, save=False, save_path=""):
        if reset:
            lsh = MinHashLSH(
                threshold=self.similarity_threshold, num_perm=self.num_perms, storage_config={
                    'type': 'cassandra',
                    'basename': b'base_lsh_cassandra',
                    'cassandra': {
                        'seeds': ['127.0.0.1', "cassandra"],
                        'keyspace': 'lsh_test',
                        'replication': {
                            'class': 'SimpleStrategy',
                            'replication_factor': '1',
                        },
                        'drop_keyspace': True,
                        'drop_tables': True,
                    }
                }
            )

        def deduplicate_wrapper(batch, indices, lsh):
            return self.hash_insert_mark_with_basename(batch, indices, lsh=lsh)

        lsh = self.get_cassandra_lsh()

        # add a duplicate column
        dataset = dataset.map(
            lambda x: self.add_is_duplicate_column(x), num_proc=cpu_count)

        # deduplicate
        dataset = dataset.map(deduplicate_wrapper, batched=True,
                              return_indices=True, num_proc=cpu_count, fn_kwargs={"lsh": lsh})

        # filter dataset
        dataset = dataset.map(
            lambda x: x["is_duplicate"] == False, num_proc=cpu_count)

        # remove dummy column
        dataset = dataset.remove_columns(["is_duplicate"])

        if save and save_path != "":
            dataset.save_to_disk(save_path)
