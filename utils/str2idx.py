class String2Index:
    def __init__(self, start_idx: int = 1) -> None:
        self._midx = dict[str, int]()
        # self._mstr = dict[int, str]()
        self._idx = start_idx

    def get_index(self, s: str):
        if s not in self._midx:
            self._midx[s] = self._idx
            self._idx += 1
        return self._midx[s]

    def get_str(self, idx: int) -> str:
        raise NotImplementedError()

    def save(self, output_path: str):
        raise NotImplementedError()

    def load(self, input_path: str):
        raise NotImplementedError()
