// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { getCore } from 'cvat-core-wrapper';

import Sam3TrackAction from './annotations-actions/sam3-track';

const core = getCore();

// Register as early as possible so it is visible in "Menu -> Run actions".
await core.actions.register(new Sam3TrackAction());

