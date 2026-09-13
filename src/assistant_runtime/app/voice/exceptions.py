"""Errors safe to expose at the voice HTTP boundary."""


class VoiceError(Exception):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code
