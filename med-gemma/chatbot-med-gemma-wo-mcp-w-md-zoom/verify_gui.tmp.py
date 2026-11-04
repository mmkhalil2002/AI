import importlib.util, sys, tempfile, itertools, time, json
from pathlib import Path
from PIL import Image
spec=importlib.util.spec_from_file_location('app', 'python/client_medgemma_patients.py')
a=importlib.util.module_from_spec(spec); sys.modules['app']=a; spec.loader.exec_module(a)
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp); patient=root/'Test-Person-01012000'; snap=patient/'2026-11-03'/'120000'
    for modality in ('xrays','ct_scan','ekgs','blood_tests'):
        (snap/modality).mkdir(parents=True)
        if modality != 'blood_tests':
            for i in range(2): Image.new('RGB',(90+i,70),'white').save(snap/modality/f'image{i}.png')
    (snap/'blood_tests'/'labs.json').write_text(json.dumps({'title':'Synthetic','results':[{'test':'A','value':12,'unit':'u','reference_range':'1-10','status':'High'}]}))
    a.patient_records_root=lambda:root
    a.refresh_models=lambda:None
    original=a.tk.Tk.mainloop
    def check(window):
        errors=[]
        window.report_callback_exception=lambda *args:errors.append(args)
        window.update()
        a.active_patient=patient; a.selected_snapshot_dir=snap
        a.load_images(); a.load_blood_tests(); a.update_selection_workspace(); window.update()
        assert len(a.galleries)==3 and len(a.thumbnail_items)==3
        menus=[]
        def walk(w):
            for c in w.winfo_children():
                if isinstance(c,a.ttk.Menubutton):menus.append(c)
                walk(c)
        walk(window)
        display=next(x for x in menus if x.cget('text')=='Display')
        menu=window.nametowidget(display.cget('menu'))
        # Exercise all sixteen display states via actual menu callbacks.
        state=[True]*4
        for combination in itertools.product((False,True),repeat=4):
            for i,wanted in enumerate(combination):
                if state[i]!=wanted:menu.invoke(i);state[i]=wanted
            window.update()
            panels=a.galleries[0]['title'].master
            expected=[a.galleries[i]['title'] for i in (0,2,1)]+[a.blood_test_panel]
            for i,(panel,visible) in enumerate(zip(expected,combination)):
                assert bool(panel.winfo_ismapped()) == visible
                if visible:
                    assert int(panel.grid_info()['row'])==i//2
                    assert int(panel.grid_info()['column'])==i%2
            if all(combination):
                assert abs(expected[0].winfo_width()-expected[1].winfo_width())<3
                assert abs(expected[0].winfo_height()-expected[2].winfo_height())<3
        for key in ('xrays','ct_scan','ekgs','blood_tests'):
            a.zoom_vars[key].set(True);a.toggle_zoom_option(key)
        deadline=time.monotonic()+1.5
        while time.monotonic()<deadline:
            window.update()
            while not a.ui_callbacks.empty():a.ui_callbacks.get_nowait()()
            time.sleep(.02)
        assert len(a.modality_viewers)==4
        assert not any(item.get('failed') for group in a.thumbnail_items for item in group)
        for key,st in a.modality_viewers.items():
            assert st['path'] is not None
            if key!='blood_tests':assert st.get('source') is not None,key
        a.xray_zoom_mode='merge';a.toggle_xray_zoom_mode()
        assert len(a.modality_viewers)==4
        a.close_all_zoom_windows();assert not a.modality_viewers
        assert not any(v.get() for v in a.zoom_vars.values())
        assert not errors,errors
        window.destroy()
    a.tk.Tk.mainloop=check
    a.main()
    a.tk.Tk.mainloop=original
    paths=tuple(snap/m/'image0.png' for m in ('xrays','ekgs','ct_scan'))
    snapshot=a.RequestSnapshot('test',patient,'http://unused','mock','Question',paths,
                               blood_paths=(snap/'blood_tests'/'labs.json',),xray_count=1,ekg_count=1,
                               user_question='Question',blood_json='[{"title":"Synthetic"}]')
    state=a.RequestState(snapshot);calls=[]; callbacks=[]
    def fake(url,model,prompt,images=(),**kw):
        calls.append((prompt,images,kw));return a.GenerationResult('Findings: mock. Limitations: mock.',{'done_reason':'stop'})
    a.generate=fake;a.set_request_stage=lambda *args:None;a.post_ui=callbacks.append
    result=a.generate_separate_reports(snapshot,state)
    assert len(calls)==4 and all(len(c[1])==1 for c in calls[:3]) and not calls[3][1]
    assert 'CT SCAN - image0' in result.answer
    assert all('Synthetic' not in c[0] for c in calls[:3])
    assert 'ct_scan' in a.report_input_reference_data(snapshot)
    calls.clear();state.cancel_event.set();assert a.generate_separate_reports(snapshot,state) is None and not calls
    a.current_request=None;a.receive_image_diagnosis('obsolete',paths[0],'stale')
    assert a.image_diagnoses.get(paths[0].resolve())!='stale'
print('PASS: syntax/import, 16 Display combinations and 2x2 grid positions/equal sizing, four independent modality viewers, async image decode, toggle cleanup, CT analysis, blood isolation, cancellation, stale response rejection')
