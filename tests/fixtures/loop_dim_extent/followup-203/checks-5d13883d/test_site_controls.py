# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metamorphic controls added after the duplicate-site failure; original probes unchanged."""
import pytest
import test_sequence_sites as s

@pytest.mark.parametrize('annotated',[False,True])
def test_distinct_loop_variable_names(tmp_path,annotated):
 name='normalized_duplicate_different_epochs' if annotated else 'duplicate_sites_different_loop_extents'
 body,count=s.VALID[name]
 body=body.replace('for e in Dim("K"):\n            A[t,e]=X[t,e]', 'for k in Dim("K"):\n            A[t,k]=X[t,k]')
 code=s.a.prog(body);rec=s.a.observe(tmp_path,code,inputs=s.a.INPUTS)
 epochs=s.record('renamed-control-'+str(annotated),code,{'observation':rec})
 assert epochs[0]==[set(),{'A'}]
 assert s.np.array_equal(s.np.asarray(rec['out']['out']),s.a.X[:,:3])
 s.a.counted(rec,count)
