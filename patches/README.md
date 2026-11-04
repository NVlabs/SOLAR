<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# patches/

Modifications to [torchview](https://github.com/mert-kurttutan/torchview)
(MIT License, Copyright (c) 2022 Mert Kurttutan) that SOLAR's graph
extraction needs. `install.sh` clones torchview at the tested commit and
applies the patches it lists; nothing here is imported by SOLAR directly.
The former `torchview-collect-attributes.patch` is gone: it was superseded by
upstream torchview #128 (commit edbe1fa, the pinned commit) and no longer
applied cleanly.

| file | purpose |
|---|---|
| `torchview-parameter-tensors.patch` | against torchview edbe1fa (0.2.7 + #128): record module parameters as graph tensors so weight reads are accounted; applied by `install.sh` |
| `torchview-collect-attributes-inputs.patch` | against the torchview 0.2.7 release: propagate `collect_attributes` to model-input and hidden RecorderTensors (the content of upstream #128, commit edbe1fa); for release installs without the pinned commit |

## Licence scope

The two `.patch` files are derivative works of torchview and are distributed
under the SOLAR project licence (Apache-2.0, see `LICENSE`) with the
torchview copyright notice retained; see the torchview entry in
`THIRD_PARTY_NOTICES`. They carry no SPDX header of their own because a
header line would alter the diff context they must apply against. Applying
a patch produces a modified torchview, which stays under the MIT License.
