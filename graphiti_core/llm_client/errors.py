"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class RateLimitError(Exception):
    """Exception raised when the rate limit is exceeded."""

    def __init__(self, message='Rate limit exceeded. Please try again later.'):
        self.message = message
        super().__init__(self.message)


class RefusalError(Exception):
    """Exception raised when the LLM refuses to generate a response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """Exception raised when the LLM returns an empty response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class TruncatedResponseError(Exception):
    """Exception raised when the LLM stopped because it hit the max_tokens cap.

    Deliberately NOT registered in ``is_server_or_retry_error``. A truncation is
    deterministic — the same prompt under the same cap produces the same clipped
    body — so retrying only reproduces it, four times, with minutes of backoff in
    between. It also has to be raised BEFORE ``json.loads``: a clipped JSON body
    surfaces as ``json.decoder.JSONDecodeError``, which IS in the retry predicate,
    so a truncation used to be misdiagnosed as a transient parse hiccup. That is
    exactly how a 4096-token cap produced 'Expecting value: line 1 column 4149'
    four times over instead of one actionable error.
    """

    def __init__(
        self,
        message: str,
        *,
        max_tokens: int | None = None,
        completion_tokens: int | None = None,
    ):
        self.message = message
        self.max_tokens = max_tokens
        self.completion_tokens = completion_tokens
        super().__init__(self.message)
