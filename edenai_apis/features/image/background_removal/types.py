import abc

from pydantic import BaseModel


class BackgroundRemovalParams(BaseModel, abc.ABC):
    pass
