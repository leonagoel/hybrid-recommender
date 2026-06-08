from threading import Lock
import time

class Node:
    def __init__(self, key: str, value: any):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class RequestLRUCache:
    def __init__(self, capacity: int = 1000):
        """
        Thread-safe, memory-bounded O(1) LRU Cache to mitigate infinite memory leaks.
        
        :param capacity: Maximum number of request entries allowed in memory.
        """
        self.capacity = capacity
        self.lock = Lock()
        
        # Fast lookup map: { key: Node }
        self.cache = {}
        
        # Doubly Linked List boundaries to track usage recency
        self.head = Node("head", None)
        self.tail = Node("tail", None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Removes an existing node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: Node):
        """Inserts a new node right behind the pseudo-head node."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: str) -> any:
        """Fetches request metadata in O(1) time and updates recency order."""
        with self.lock:
            if key not in self.cache:
                return None
            
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)  # Move to head because it was recently accessed
            return node.value

    def put(self, key: str, value: any):
        """Caches request payload metadata in O(1) time with proactive eviction."""
        with self.lock:
            if key in self.cache:
                # Update existing value and move to head
                node = self.cache[key]
                node.value = value
                self._remove(node)
                self._add_to_head(node)
            else:
                new_node = Node(key, value)
                self.cache[key] = new_node
                self._add_to_head(new_node)
                
                # Eviction: If capacity is exceeded, evict the tail node (Least Recently Used)
                if len(self.cache) > self.capacity:
                    lru_node = self.tail.prev
                    self._remove(lru_node)
                    del self.cache[lru_node.key]  # Safely free up system memory leak track

# ========================================================
# Operational Verification Script
# ========================================================
if __name__ == "__main__":
    # Test bounded capacity of 2 items
    cache = RequestLRUCache(capacity=2)
    
    print("--- Storing Request Metadata (Keys: req_1, req_2) ---")
    cache.put("req_1", "Recommendation Payload A")
    cache.put("req_2", "Recommendation Payload B")
    
    print(f"Fetch req_1: {cache.get('req_1')} (Now active most-recent)")
    
    print("\n--- Inserting req_3 (Triggers eviction of req_2) ---")
    cache.put("req_3", "Recommendation Payload C")
    
    print(f"Fetch req_2 (Should be Evicted): {cache.get('req_2')}")
    print(f"Fetch req_1 (Should still Exist): {cache.get('req_1')}")