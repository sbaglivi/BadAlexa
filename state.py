from dataclasses import dataclass

Message= tuple[str, str]

class AppState: ...

@dataclass
class Idle(AppState): ...

@dataclass
class Listening(AppState):
    prev_turns: list[Message]
    last_voiced_ts: float | None
    start_ts: float

@dataclass
class Generating(AppState): 
    prev_turns: list[Message]