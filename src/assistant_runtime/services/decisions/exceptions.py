"""Errors safe to expose at the decisions HTTP boundary."""


class DecisionError(Exception):
    """A decision call that did not produce answers, with the HTTP status it maps to.

    ``provider_status_code`` is the provider's HTTP status when the provider
    answered with an error, and ``provider_detail`` the bounded reason it gave.
    The state and the key never travel on this exception.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        provider_status_code: int | None = None,
        provider_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_status_code = provider_status_code
        self.provider_detail = provider_detail

    @property
    def metadata(self) -> dict[str, int | str]:
        return {
            key: value
            for key, value in {
                "provider_status_code": self.provider_status_code,
                "provider_detail": self.provider_detail,
            }.items()
            if value is not None
        }
