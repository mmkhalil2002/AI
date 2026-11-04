from pathlib import Path
p=Path('python/client_medgemma_patients.py')
s=p.read_text(encoding='utf-8')
s=s.replace('data_panes = ttk.Panedwindow(body, orient=tk.VERTICAL)', 'data_panes = ttk.Frame(body, style="App.TFrame")')
s=s.replace('        data_panes.add(pane, weight=1)', '        # Record panels are positioned together after construction.')
s=s.replace('    data_panes.add(blood_test_panel, weight=1)', '    # Blood Test occupies the lower-right record cell.')
a=s.index('    def equalize_record_panels(event=None):');b=s.index('        for index, (label, visible)',a)
s=s[:a]+'''    def equalize_record_panels(event=None):
        # Grid weights keep both rows and columns equal on window resize.
        for index in range(2):
            data_panes.rowconfigure(index, weight=1, uniform="record_rows")
            data_panes.columnconfigure(index, weight=1, uniform="record_columns")

    def update_record_panel_visibility():
        # Preserve modality positions when an individual panel is hidden.
        for index, (panel, visible) in enumerate(zip(record_panels, panel_visibility)):
            if visible.get():
                panel.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
            else:
                panel.grid_remove()
'''+s[b:]
# CT uses the same safe thumbnail dimensions as X-ray.
s=s.replace('    (image_dimension("EKG_WIDTH"), image_dimension("EKG_HEIGHT")),\n)', '    (image_dimension("EKG_WIDTH"), image_dimension("EKG_HEIGHT")),\n    (image_dimension("XRAY_WIDTH"), image_dimension("XRAY_HEIGHT")),\n)')
s=s.replace('        if error_name or picture is None or not item["button"].winfo_exists():\n            item["failed"] = True', '        if not item["button"].winfo_exists():\n            continue\n        if error_name or picture is None:\n            item["failed"] = True')
p.write_text(s,encoding='utf-8')
import ast
ast.parse(s)
print('Syntax OK; 2x2 grid applied')
