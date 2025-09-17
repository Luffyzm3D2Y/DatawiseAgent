# Copyright (c) 2024, Owners of https://github.com/Luffyzm3D2Y/DatawiseAgent
# SPDX-License-Identifier: Apache-2.0
#
# This file includes modifications based on code from:
#   https://github.com/microsoft/autogen
#   Copyright (c) Microsoft Corporation
#   SPDX-License-Identifier: MIT
#
# Substantial modifications and new features have been added.
from typing import Dict, List, Literal, TypedDict, Union

MessageContentType = Union[str, List[Union[Dict, str]], None]


class UserMessageTextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class UserMessageImageContentPart(TypedDict):
    type: Literal["image_url"]
    # Ignoring the other "detail param for now"
    image_url: Dict[Literal["url"], str]
