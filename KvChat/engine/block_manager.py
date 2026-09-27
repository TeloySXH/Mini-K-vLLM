from collections import deque
import xxhash
import numpy as np
import logging
import threading

from KvChat.engine.sequence import Sequence

logger = logging.getLogger(__name__)

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_collision_detection=True):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        
        # [OPTIMIZATION] Enable hash collision detection and statistics
        self.enable_collision_detection = enable_collision_detection
        self.collision_stats = {
            'collisions': 0,
            'total_hashes': 0,
        }

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0

        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        collision_count = 0  # [OPTIMIZATION] Count collisions
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            block_id = self.hash_to_block_id.get(h, -1)
            self.collision_stats['total_hashes'] += 1  # [OPTIMIZATION] Track hash count
            
            # [OPTIMIZATION] Triple verification: hash value + token_ids + ref_count
            if block_id != -1:
                block = self.blocks[block_id]
                
                # Verify 1: hash value matches
                if block.hash == h:
                    # Verify 2: token_ids match
                    if block.token_ids == token_ids:
                        # Cache hit confirmed
                        seq.num_cached_tokens += self.block_size
                        block.ref_count += 1
                        cache_miss = False
                    else:
                        # Hash collision detected!
                        collision_count += 1
                        if self.enable_collision_detection:
                            logger.warning(
                                f"Hash collision detected! Hash={h}, "
                                f"existing_tokens_len={len(block.token_ids)}, "
                                f"new_tokens_len={len(token_ids)}"
                            )
                        cache_miss = True
                else:
                    # Hash value mismatch (content may have been overwritten)
                    cache_miss = True
            else:
                # Hash not found
                cache_miss = True
            
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)
        
        # [OPTIMIZATION] Log collision statistics
        if collision_count > 0:
            self.collision_stats['collisions'] += collision_count
            logger.info(
                f"Allocation: {collision_count} collisions detected in seq {seq.seq_id}. "
                f"Total collisions: {self.collision_stats['collisions']}/{self.collision_stats['total_hashes']}"
            )
            
            
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    def truncate_blocks(self, seq: Sequence, keep_blocks: int):
        """
        Release blocks in seq.block_table beyond keep_blocks.
        keep_blocks: number of blocks to keep (e.g. N+1)
        """
        if keep_blocks >= len(seq.block_table):
            return
        # free tail blocks
        tail = seq.block_table[keep_blocks:]
        for block_id in reversed(tail):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # trim seq.block_table and adjust num_cached_tokens
        seq.block_table = seq.block_table[:keep_blocks]
        seq.num_cached_tokens = min(seq.num_cached_tokens, keep_blocks * self.block_size)
        # now process the new last block
        if seq.block_table:
            last_block_id = seq.block_table[-1]
            last_block = self.blocks[last_block_id]
            #the new last block now just cached one token(according to our compress trigger mechanism)
            #and maybe this block is  full before compress, so we have to change its hash to -1
            if last_block.hash != -1:
                last_block.hash = -1
        