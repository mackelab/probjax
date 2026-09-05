import heapq
from typing import Any, List, Union


class PriorityQueue:
    heap: List[tuple[Union[int, float], int, Any]]  # Heap data structure
    entry_finder: dict[Any, tuple[int, Union[int, float]]]
    counter: int  # Unique sequence count -> Queue position

    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0

    def __repr__(self) -> str:
        return str([(element, data[1]) for element, data in self.entry_finder.items()])

    __str__ = __repr__

    def __contains__(self, element) -> bool:
        return element in self.entry_finder

    def __len__(self) -> int:
        return len(self.entry_finder)

    def insert(self, element: Any, cost: Union[int, float] = 0.0):
        """This function will insert an element into the priority queue.

        Args:
            element (Any): Element to be inserted into the priority queue.
            cost (Number): Cost of the element. Defaults to 0.0.
        """
        token = self.counter
        entry = (cost, token, element)
        self.entry_finder[element] = (token, cost)
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def append(self, element: Any, cost: Union[int, float] = 0.0):
        self.insert(element, cost)

    def pop(self):
        """This function will pop the element with the lowest cost or which is longest
        in the queue (if costs are equal)."""
        while self.heap:
            cost, token, element = heapq.heappop(self.heap)
            del cost
            current = self.entry_finder.get(element)
            if current is None:
                continue
            current_token, _ = current
            if token != current_token:
                continue

            del self.entry_finder[element]
            return element
        raise IndexError("Priority queue is empty.")

    def update_cost(self, element: Any, new_cost: Union[int, float]):
        """This function does update the cost of a given element.

        Args:
            element (Any): Element which cost is to be updated.
            new_cost (Union[int, float]): New cost of the element.
        """
        if element in self.entry_finder:
            token = self.counter
            self.counter += 1
            self.entry_finder[element] = (token, new_cost)
            heapq.heappush(self.heap, (new_cost, token, element))

    def is_empty(self):
        """This function checks if the priority queue is empty."""
        return len(self.entry_finder) == 0
