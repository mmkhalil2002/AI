from pathlib import Path
p=Path('verify_gui.tmp.py');s=p.read_text()
a=s.index('            assert len(panels.panes())');b=s.index('        for key in',a)
s=s[:a]+'''            expected=[a.galleries[i]['title'] for i in (0,2,1)]+[a.blood_test_panel]
            for i,(panel,visible) in enumerate(zip(expected,combination)):
                assert bool(panel.winfo_ismapped()) == visible
                if visible:
                    assert int(panel.grid_info()['row'])==i//2
                    assert int(panel.grid_info()['column'])==i%2
            if all(combination):
                assert abs(expected[0].winfo_width()-expected[1].winfo_width())<3
                assert abs(expected[0].winfo_height()-expected[2].winfo_height())<3
'''+s[b:]
s=s.replace('16 Display combinations and equal sizing','16 Display combinations and 2x2 grid positions/equal sizing')
s=s.replace("        assert len(a.modality_viewers)==4\n        for key,st", "        assert len(a.modality_viewers)==4\n        assert not any(item.get('failed') for group in a.thumbnail_items for item in group)\n        for key,st")
p.write_text(s)
