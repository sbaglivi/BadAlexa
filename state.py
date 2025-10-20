from dataclasses import dataclass

class AppState: ...

@dataclass
class Idle(AppState): ...

@dataclass
class Listening(AppState):
    last_voiced_ts: float | None
    start_ts: float

@dataclass
class Generating(AppState): ...