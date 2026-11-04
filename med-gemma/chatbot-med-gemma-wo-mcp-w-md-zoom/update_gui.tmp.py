from pathlib import Path
p=Path('python/client_medgemma_patients.py'); s=p.read_text(encoding='utf-8')
s=s.replace('    completion: dict = field(default_factory=dict)','    completion: dict = field(default_factory=dict)\n    image_bytes: dict = field(default_factory=dict)')
s=s.replace('        models = list_models(snapshot.url)','        state.image_bytes = {Path(path): Path(path).read_bytes() for path in snapshot.image_paths}\n        models = list_models(snapshot.url)')
s=s.replace('payload["images"] = [encode_image(path) for path in image_paths]', 'payload["images"] = [base64.b64encode(request_state.image_bytes[Path(path)]).decode("ascii")\n                             if request_state and Path(path) in request_state.image_bytes else encode_image(path)\n                             for path in image_paths]')
s=s.replace('("Blood test", snapshot.blood_paths),','("CT Scan", snapshot.image_paths[xray_count + ekg_count:]),\n        ("Blood test", snapshot.blood_paths),')
s=s.replace('"blood_test": relative_refs(snapshot.blood_paths),','"ct_scan": relative_refs(images[xray_count + ekg_count:]),\n        "blood_test": relative_refs(snapshot.blood_paths),')
s=s.replace('(("X-ray", "xray"), ("EKG", "ekg"), ("Blood test", "blood_test"))','(("X-ray", "xray"), ("CT Scan", "ct_scan"), ("EKG", "ekg"), ("Blood test", "blood_test"))')
s=s.replace('refs = references[key]', 'refs = references.get(key, [])')
s=s.replace('viewer.protocol("WM_DELETE_WINDOW", lambda p=path: close_xray_zoom_viewer(p))','viewer.protocol("WM_DELETE_WINDOW", lambda p=path: user_close_selection(p))')
s=s.replace('viewer.protocol("WM_DELETE_WINDOW", close_image_viewer)', 'viewer.protocol("WM_DELETE_WINDOW", user_close_merged_selection)')
pos=s.index('\nmodality_viewers = {}')
s=s[:pos]+'''
def user_close_selection(path):
    global xray_zoom_enabled
    close_xray_zoom_viewer(path)
    if not xray_zoom_viewers and image_viewer is None:
        xray_zoom_enabled = False
        zoom_vars["selection"].set(False)


def user_close_merged_selection():
    global xray_zoom_enabled
    close_image_viewer()
    if not xray_zoom_viewers:
        xray_zoom_enabled = False
        zoom_vars["selection"].set(False)

'''+s[pos:]
s=s.replace('    state["zoom"] = 1.0\n    state["title"].config', '    canvas = state["canvas"]\n    fit = viewer_fit_scale(source.size, (max(1, canvas.winfo_width()), max(1, canvas.winfo_height())))\n    state["zoom"] = 1.0 / max(fit, 0.0001)\n    state["window"].title(f"{modality} - {path.stem} - {source.width}x{source.height}")\n    state["title"].config')
s=s.replace('    viewer.title("Image detail viewer")', '    viewer.title("Image detail viewer")\n    viewer.maxsize(1400, 900)')
s=s.replace('font=("Segoe UI", 18, "bold"), anchor="w", padx=12, pady=10)', 'font=("Segoe UI", 18, "bold"), anchor="w", padx=12, pady=10)')
p.write_text(s,encoding='utf-8')
import ast
ast.parse(s)
print('Syntax OK')
