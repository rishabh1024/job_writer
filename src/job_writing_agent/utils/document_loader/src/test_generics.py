from typing import Generic, Sequence, TypeVar, reveal_type

K = TypeVar("K")

V = TypeVar("V")

T = TypeVar("T")


def get_first_element(collection: Sequence[T]) -> T:
    return collection[0]


integer_element = get_first_element([1, 2, 3])
string_element = get_first_element(["a", "b", "c"])

print(reveal_type(integer_element))
print(reveal_type(string_element))


_map = {"Rishabh": 30, "Aggarwal": 33, "Guitar": 10}


def get_value(dict_map: dict[K, V], key: K) -> V:
    return dict_map[key]


print(get_value(_map, "Rishabh"))
print(reveal_type(get_value(_map, "Rishabh")))


# Generics in classes


class CustomList(Generic[T]):
    def __init__(self, items: list[T]):
        self.items = items

    def append(self, item: T):
        self.items.append(item)

    def get_items(self) -> list[T]:
        return self.items

    def remove_item(self, item: T):
        if item in self.items:
            self.items.remove(item)


custom_list = CustomList([1, 2, 3])

for item in custom_list.get_items():
    print(item)


custom_list = CustomList(["a", "b", "c"])

for item in custom_list.get_items():
    print(item)
