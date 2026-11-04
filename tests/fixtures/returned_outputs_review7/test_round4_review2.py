"""Independent lattice/set oracle controls; all capture and tensor work is CPU-only."""
from pathlib import Path as P
import json,itertools,random,math,time,copy
import pytest,torch,yaml
from solar.einsum.pytorch_to_einsum import _returned_leaf_region as region,_returned_group_footprint as footprint,_lattice_rects
from test_review7_aliases import capture,convert
N=(1<<20)+1

def r(offset,shape,stride,path,width=4,size=None):
    x=region({'storage_offset':offset,'shape':shape,'stride':stride,'element_size':width,'storage_nbytes':size},{})
    x['path']=path;return x
def save(tmp_path,name,data):
    (tmp_path/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')

def test_three_large_stride3_views_are_disjoint(tmp_path):
    a,b,c=[r(i,[N],[3],str(i)) for i in range(3)]
    fp=footprint([b,c],[a],12*N);whole=footprint([a,b,c],[],12*N)
    save(tmp_path,'RESULT',{'remainder':fp,'whole':whole})
    assert fp['union_bytes']==8*N and fp['covered_bytes']==0 and fp['period']==12
    assert whole['union_bytes']==12*N and whole['unbound_exact']

def test_overlapping_periods_use_gcd_lcm_grid(tmp_path):
    a=r(0,[N],[6],'a');b=r(0,[N],[10],'b');common=(N-1)//5+1
    fp=footprint([a],[b],None);save(tmp_path,'RESULT',fp)
    assert fp['period']==120 and fp['covered_bytes']==4*common and fp['union_bytes']==4*(N-common)
    assert fp['unbound_exact'] and fp['coverage_exact']

def test_large_lattice_interval_partial_overlap(tmp_path):
    fp=footprint([r(0,[N],[3],'u')],[r(2,[9],[1],'b')],12*N)
    save(tmp_path,'RESULT',fp);assert fp['covered_bytes']==12 and fp['union_bytes']==4*(N-3)

def test_signed_lattice_matches_positive_metadata(tmp_path):
    neg=r(3*(N-1),[N],[-3],'negative');pos=r(0,[N],[3],'positive')
    fp=footprint([neg],[pos],12*N);save(tmp_path,'RESULT',{'negative':neg,'positive':pos,'footprint':fp,'evidence_kind':'serialized negative-stride metadata; not a live Torch negative-stride tensor'})
    assert neg['lattice']==pos['lattice'] and fp['union_bytes']==0 and fp['covered_bytes']==4*N and fp['unbound_exact'] and fp['coverage_exact']

@pytest.mark.parametrize('count',[9,N])
def test_broadcast_base_coverage_and_diagnostic(tmp_path,count):
    view=r(0,[count],[0],'broadcast');base=r(0,[1],[1],'base')
    fp=footprint([view],[base],4);save(tmp_path,'RESULT',{'region':view,'footprint':fp,'exact_unique_bytes':4})
    assert fp['union_bytes']==0 and fp['covered_bytes']==4 and fp['unbound_exact'] and fp['coverage_exact']

@pytest.mark.parametrize('case',['two_dim','three_dim','zero_plus_outer'])
def test_multi_outer_regions_above_cap_are_approximate_never_cover(tmp_path,case):
    shape,stride={'two_dim':([1025,1025],[4096,2]),'three_dim':([129,129,129],[100000,258,2]),'zero_plus_outer':([N,2],[2,0])}[case]
    bound=r(0,shape,stride,'approximate-bound');assert not bound['exact']
    point=r(1,[1],[1],'hole');fp=footprint([point],[bound],None)
    unbound=footprint([bound],[],None)
    save(tmp_path,'RESULT',{'region':bound,'bound_case':fp,'unbound_case':unbound})
    assert fp['union_bytes']==4 and fp['covered_bytes']==0 and not fp['coverage_exact'] and fp['unproven_bound']==['approximate-bound']
    assert unbound['union_bytes']==bound['bytes'] and not unbound['unbound_exact']

@pytest.mark.parametrize('strides',[(64,1),(65,1),(67,71),(1000000007,1000000009)])
def test_period_split_cap_is_bounded_and_fails_closed(tmp_path,strides):
    # Keep both operands as explicit exact lattices: stride 1 is ordinarily a contiguous interval.
    def lat(st,path):return {'intervals':None,'lattice':(0,st,10**30,1),'exact':True,'method':'strided_lattice_exact','bytes':10**30,'path':path}
    a,b=[lat(st,p) for st,p in zip(strides,['unbound','bound'])];t=time.monotonic();fp=footprint([a],[b],None)
    p=math.lcm(*strides);counts=[None if _lattice_rects(x['lattice'],p) is None else len(_lattice_rects(x['lattice'],p)) for x in [a,b]]
    save(tmp_path,'RESULT',{'footprint':fp,'rectangle_counts':counts,'seconds':time.monotonic()-t})
    assert all(n is None or n<=128 for n in counts)
    if max(p//st for st in strides)>64:
        assert fp['approximate_unbound'] or fp['unproven_bound']
        assert not (fp['unbound_exact'] and fp['coverage_exact'])
        if counts[1] is None:assert fp['covered_bytes']==0 and fp['union_bytes']>0
    else:assert fp['unbound_exact'] and fp['coverage_exact']

def test_lattice_group_against_independent_small_byte_oracle(tmp_path):
    rng=random.Random(12004);exact=approx=0
    def make(path):
        if rng.randrange(4)==0:
            start=rng.randrange(100);end=start+rng.randrange(1,20)
            return {'intervals':[(start,end)],'lattice':None,'exact':True,'method':'byte_region_union','bytes':end-start,'path':path},set(range(start,end))
        st=rng.randrange(1,20);run=rng.randrange(1,st+1);start=rng.randrange(40);count=rng.randrange(1,20)
        return {'intervals':None,'lattice':(start,st,count,run),'exact':True,'method':'strided_lattice_exact','bytes':count*run,'path':path},{start+k*st+i for k in range(count) for i in range(run)}
    for i in range(500):
        ua=[make('u'+str(j)) for j in range(rng.randrange(1,4))];ba=[make('b'+str(j)) for j in range(rng.randrange(4))]
        us=set().union(*(x[1] for x in ua));bs=set().union(*(x[1] for x in ba))
        fp=footprint([x[0] for x in ua],[x[0] for x in ba],None);truth=len(us-bs)
        if fp['unbound_exact'] and fp['coverage_exact']:
            assert fp['union_bytes']==truth and fp['covered_bytes']==len(us&bs),(i,ua,ba,fp,truth);exact+=1
        else:assert fp['union_bytes']>=truth,(i,ua,ba,fp,truth);approx+=1
    save(tmp_path,'RESULT',{'seed':12004,'cases':500,'exact':exact,'conservative_fallback':approx})

def test_real_irregular_index_select_is_a_distinct_contiguous_allocation(tmp_path):
    base=torch.arange(2*N,dtype=torch.float32)
    index=torch.cat([torch.arange(0,2*N,4),torch.arange(2,2*N,4).flip(0)])
    out=torch.index_select(base,0,index)
    assert out.numel()==N and out.untyped_storage()._cdata!=base.untyped_storage()._cdata and out.is_contiguous()
    rr=r(out.storage_offset(),list(out.shape),list(out.stride()),'index_select',out.element_size(),out.untyped_storage().nbytes())
    save(tmp_path,'RESULT',{'shape':list(out.shape),'stride':list(out.stride()),'distinct_storage':True,'region':rr,'interpretation':'index_select copies; index irregularity is not irregular byte layout of a storage-sharing view'})
    assert rr['exact'] and rr['method']=='byte_region_union' and rr['bytes']==4*N

@pytest.mark.parametrize('expression,expected',[
    ('y, y[:1].expand(17).as_subclass(torch.Tensor)',32),
    ('y[::3], y[1::3].as_subclass(torch.Tensor), y[2::3].as_subclass(torch.Tensor)',32),
])
def test_live_small_storage_views_keep_exact_union(tmp_path,expression,expected):
    pg,oracle,meta=capture(tmp_path,expression);eg,total=convert(tmp_path,pg)
    assert oracle==expected and total['external_output_bytes']==expected

def test_approximate_bound_cannot_establish_alias_in_converter(tmp_path):
    # Explicit serialized-metadata control on a captured skeleton, not a claimed live large view.
    pg,_,_=capture(tmp_path,'y[::2], y[1:2].as_subclass(torch.Tensor)')
    b=pg['returned_outputs'][0];ident=b['tensor_identity'];ident.update(shape=[1025,1025],stride=[4096,2],storage_offset=0,storage_nbytes=20_000_000,numel=1025**2)
    b.update(shape=[1025,1025],nbytes=4*1025**2)
    pg['returned_outputs'][1]['tensor_identity']['storage_nbytes']=20_000_000
    eg,total=convert(tmp_path,pg);fp=next(v for v in eg['returned_footprint'].values() if v.get('leaves'))
    assert fp['union_bytes']==4 and fp['covered_by_bound_bytes']==0 and fp['unproven_bound'] and fp['method']!='alias_of_bound_output'
    assert total['status']=='incomplete' and 'returned_output_unbound' in total['incomplete_layers'].values()
