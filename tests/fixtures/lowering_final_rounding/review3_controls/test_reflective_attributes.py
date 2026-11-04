import importlib.util,pathlib,pytest,numpy as np
R=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location("independent",R/"test_independent_round7.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.OUT=R/"runs/reflective-attributes";p.OUT.mkdir(parents=True,exist_ok=True)
BASE='import numpy as np\n'
ATTR='np.ma.core.operator.attrgetter(b"__globals__".decode())(seed)'
CHR='np.ma.core.operator.attrgetter(chr(95)*2+"globals"+chr(95)*2)(seed)'
CASES={
"library_builtin_exec":BASE+'def marker(arg=np.ma.core.builtins.exec("DT = \'fp32\'")):\n    pass\n',
"library_builtin_getattr":BASE+'def seed():\n    pass\ndef marker(arg=np.ma.core.builtins.getattr(seed,b"__globals__".decode()).update({"DT":"fp32"})):\n    pass\n',
"operator_bytes_attrgetter":BASE+'def seed():\n    pass\n'+f'def marker(arg={ATTR}.update({{"DT":"fp32"}})):\n    pass\n',
"operator_chr_attrgetter":BASE+'def seed():\n    pass\n'+f'def marker(arg={CHR}.update({{"DT":"fp32"}})):\n    pass\n',
"operator_kernel_object_replaced":BASE+'def seed():\n    pass\n'+f'def marker(arg={ATTR}.update({{"kernel":lambda f:f}})):\n    pass\n',
}
@pytest.mark.parametrize("fp64",[True,False])
@pytest.mark.parametrize("name",list(CASES))
def test_indirect_reflective_default_cannot_misround(name,fp64):
 r=p.observe(name,CASES[name],fp64)
 assert "error" not in r,r
 assert r["public"]==r["slow"],r
 assert not r["lowered"] and r["declines"][0].startswith("kernel_output_dtype_unestablished:k"),r

