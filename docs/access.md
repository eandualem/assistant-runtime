# Identity and access

Who a caller is, what they may reach, and how the server is meant to be
exposed. The runtime does not ship an account system: it accepts a
**trusted principal** from the host and applies the same ownership and
administration rules on HTTP, Socket.IO and in-process calls.

## The exposure model

`ACCESS__MODE` picks how a principal is established:

| Mode | Who the caller is | Intended exposure |
|---|---|---|
| `trusted_local` (default) | every caller is the local operator, an administrator | the server bound to localhost (the default), used by one person or one trusted process |
| `header` | the principal id in `X-Assistant-Principal`, roles in `X-Assistant-Roles` (comma-separated; `admin` administers) | behind a reverse proxy that authenticates callers and sets those headers; the server must not be reachable any other way, since anything that can reach it can set the headers |
| `host` | whatever `AssistantDefinition.authenticate(credentials)` returns | a Python host with its own identity system (tokens, cookies, sessions) |

Header names are configurable (`ACCESS__PRINCIPAL_HEADER`,
`ACCESS__ROLES_HEADER`). A caller that cannot be identified gets `401`
(HTTP) or is refused at connect time (Socket.IO). Nothing a client puts in
a message body changes who it is: `AssistantRequest` carries no identity
fields, and any it invents are ignored or rejected.

`ACCESS__CORS_ORIGINS` lists the allowed browser origins; the default `*`
suits the localhost installation and should be restricted whenever the
server is exposed (credentials are only allowed with explicit origins).

## The host callback

```python
from assistant_runtime.main import AssistantDefinition
from assistant_runtime.principal import Credentials, Principal


async def authenticate(credentials: Credentials) -> Principal | None:
    scheme, _, token = (credentials.header("authorization") or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None                                        # -> 401 / connection refused
    user = await my_identity_system.verify(token.strip())  # the host's own system
    if user is None:
        return None
    return Principal(id=user.id, roles=set(user.roles), label=user.name)


assistant = AssistantDefinition(authenticate=authenticate)
```

`Credentials` carries the transport (`http`, `socketio`, `in_process`),
the lower-cased headers, the Socket.IO connect `auth` payload and the
client address. The callback may be sync or async; raising
`AuthenticationError` gives the caller that message, any other exception
is logged and reported as a plain authentication failure.

## Ownership

A session belongs to the principal whose message created it
(`owner_id`, stored on the row and in memory). A principal may read,
join, message, steer, continue, cancel, repair and delete its own
sessions and nothing else: another principal's session answers `403`
on HTTP, `assistant:error` of type `forbidden` on Socket.IO, and a
terminal `final_response` / `error` of type `forbidden` for a streamed
turn. Administrators reach every session and `GET /api/sessions` lists
all of them for an administrator and only the caller's own otherwise.

Sessions without an owner — rows created before ownership existed — are
reachable only by administrators until one assigns them with
`PATCH /api/sessions/{id}/owner {"owner_id": "..."}` (null makes a session
unowned again, i.e. administrator-only). Migration `0021` adds the column
and leaves existing rows unowned; assign them when moving a shared
installation out of `trusted_local`. The owner is written when the session
is created and changed only through that route; the state saves a turn
performs never touch it.

In-process callers (`create_runtime`, the CLI, the ingress) act as the
local operator unless they pass `principal=` to `run_message`,
`stream_message`, `cancel_session`, `accept_steering` or `warm_session`.

## Administration

Ordinary use is chatting, reading one's own sessions, and reading
settings, models and the artifact profile. Everything that changes the
installation is administration and requires the `admin` role: writing
settings, provider keys and OAuth, injecting messages and the inbox,
debugging routes, and every artifact mutation (propose, update, approve,
rollback, delete) and session reassignment.

## Inside a turn

Tools see the trusted principal through
`assistant_runtime.services.tools._request_context.get_current_principal()`;
artifact policies are evaluated for the `assistant` actor, so a model
cannot escalate through a tool either. Host context and attachments are
application context only; see [the host contract](host-contract.md).
