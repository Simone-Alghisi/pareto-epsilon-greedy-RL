from poke_env.environment.move import Move as OriginalMove
from poke_env.data import MOVES
from typing import Optional

class Move(OriginalMove):
    r"""
    Original poke-env move just with an equal and hash method
    """
    def __init__(self, move_id: str, raw_id: Optional[str] = None) -> None:
        super(Move, self).__init__(move_id, raw_id)
    
    def __eq__(self, o: object) -> bool:
        r"""
        Two moves are equal only if their name is equal
        Args:
          - o: object, object to compare
        Returns:
          equals: bool, whether they are equal
        """
        return self._id == o._id

    def __hash__(self) -> int:
        r"""
        Hash method
        Returns:
          hash: int, hash of the move_name
        """
        return hash(self._id)

    
    def get_showdown_name(self) -> str:
      return MOVES[self._id]["name"]
