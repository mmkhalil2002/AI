"""Exercise arbitrary section names and 1–4 window layouts using synthetic records."""
from pathlib import Path
import importlib
import json
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from record_config import parse_window_server_map

TABLE = 'Lab Results:ID004,Chest Images:ID001,Heart Trace:ID003'
mapping = parse_window_server_map(TABLE)
assert list(mapping) == ['blood_tests', 'xrays', 'ekgs']
assert parse_window_server_map('anything I choose:ID002')['ct_scan'][0] == 'anything I choose'
for invalid in ('../bad:ID001', 'Same:ID001,same:ID002', 'One:ID001,Two:ID001', 'Unknown:ID999', ''):
    try:
        parse_window_server_map(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Accepted invalid configuration: {invalid}')

with tempfile.TemporaryDirectory(prefix='window-mapping-') as temp, patch.dict(os.environ, {'WINDOW_SERVER_MAP': TABLE}):
    root = Path(temp)
    patient = root / 'SyntheticPatient'
    for name, filename, content in [('Chest Images','chest.png',b'synthetic-image'), ('Heart Trace','trace.png',b'synthetic-trace'), ('Lab Results','lab.json',b'{"results":[{"test":"Example","value":1}]}')]:
        folder = patient / name
        folder.mkdir(parents=True)
        (folder / filename).write_bytes(content)
    (patient / 'Chest Images' / 'sources.csv').write_text('excluded')
    from mcp_client import MCPRecords
    client = MCPRecords(root)
    try:
        assert set(client.sessions) == {'ID001','ID003','ID004'}
        assert client.list_records('SyntheticPatient/Chest Images') == ['SyntheticPatient/Chest Images/chest.png']
        assert client.read_record_bytes('SyntheticPatient/Heart Trace/trace.png') == b'synthetic-trace'
        assert client.read_blood_record('SyntheticPatient/Lab Results/lab.json')['results'][0]['value'] == 1
        assert client.call('ID001','list_records',{'relative_folder':'SyntheticPatient/Chest Images'})['paths']
        # A server cannot read another configured section.
        try:
            client.call('ID001','read_record',{'relative_path':'SyntheticPatient/Heart Trace/trace.png'})
        except Exception:
            pass
        else:
            raise AssertionError('Cross-section access allowed')
        print('PASS: three named MCP identities; arbitrary folder routing; image/JSON reads; section confinement')
    finally:
        client.close()

    import client_medgemma_patients_mcp as gui
    import tkinter as tk
    real_tk = tk.Tk
    class HiddenTk(real_tk):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.withdraw()
    for table in ('Only Heart:ID003', 'Scan:ID002,Labs:ID004', TABLE, TABLE+',Scan:ID002'):
        current = parse_window_server_map(table)
        gui.WINDOW_SERVER_MAP = current
        gui.CONFIGURED_INDICES = tuple(gui.RECORD_TYPES.index(key) for key in current)
        gui.MODALITY_LABELS = tuple(current.get(key, ('',''))[0] for key in gui.RECORD_TYPES)
        gui.GALLERY_NAMES = gui.MODALITY_LABELS[:3]
        gui.modality_window_enabled = [key in current for key in gui.RECORD_TYPES]
        with patch.object(gui,'patient_records_root',return_value=root), patch.object(gui,'discover_patients',return_value=[]), patch.object(tk,'Tk',HiddenTk), patch.object(HiddenTk,'mainloop',lambda self: None):
            gui.main()
            try:
                assert sum(button is not None for button in gui.modality_zoom_buttons) == len(current)
                assert sum(button is not None for button in gui.modality_window_buttons) == len(current)
                panels = [gallery['title'] for gallery in gui.galleries] + [gui.blood_test_panel]
                assert sum(panel.winfo_manager() == 'grid' for panel in panels) == len(current)
                for index in gui.CONFIGURED_INDICES:
                    assert gui.MODALITY_LABELS[index] in gui.modality_zoom_buttons[index].cget('text')
                    gui.modality_window_buttons[index].invoke()
                    assert panels[index].winfo_manager() == ''
                    gui.modality_window_buttons[index].invoke()
                    assert panels[index].winfo_manager() == 'grid'
                print(f'PASS: {len(current)} windows produce matching Zoom/ON-OFF controls and panels; toggles collapse/restore')
            finally:
                gui.window.destroy()
print('All mapping checks passed (synthetic records only; no inference).')
