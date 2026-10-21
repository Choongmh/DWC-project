"""Read-only checks of the assembled tables, forecast and delivered workbook."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import openpyxl

root=Path(__file__).resolve().parents[1]
master=pd.read_parquet(root/'data/processed/master_table.parquet')
assert master.shape==(100,129)
assert master['case'].is_unique
assert master['split'].value_counts().to_dict()=={'train':70,'validation':15,'test':15}
fc=pd.read_csv(root/'outputs/t3_forecast_all.csv',parse_dates=['date'])
assert not fc.duplicated(['id','scenario','date']).any()
main=fc[fc['scenario'].eq('main') & fc['date'].gt('2008-01-01')]
dates=pd.date_range('2008-04-01','2028-01-01',freq='QS')
wb=openpyxl.load_workbook(root/'outputs/09 Template Deliverable_filled.xlsx',read_only=True,data_only=True)
checked=0
for case in range(1,11):
    ws=wb[f'Case {case}']
    records=list(ws.iter_rows(min_row=2,max_col=4,values_only=True))
    assert len(records)==80
    actual_dates=pd.to_datetime([r[0] for r in records],format='%m/%d/%Y')
    assert (actual_dates==dates).all()
    values=np.array([r[1:] for r in records],dtype=float)
    expected=main[main['id'].eq(case)].sort_values('date')[['gas_cum','oil_cum','water_cum']].to_numpy()
    assert np.isfinite(values).all() and (values>=0).all()
    assert (np.diff(values,axis=0)>=-1e-8).all()
    np.testing.assert_allclose(values,expected,rtol=1e-12,atol=1e-6)
    checked+=values.size
wb.close()
report={'master_shape':list(master.shape),'forecast_cells_verified':checked,
        'date_count_per_case':80,'last_date':'2028-01-01',
        'template_sha256':hashlib.sha256((root/'outputs/09 Template Deliverable_filled.xlsx').read_bytes()).hexdigest(),
        'remote_verified':False,'pbix_open_refresh_verified':False,
        'docx_visual_render_verified':False}
(root/'audit/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
