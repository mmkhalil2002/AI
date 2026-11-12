"""Check connection transitions and response cancellation without network or patient data."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading

source = Path(__file__).resolve().parents[1] / 'python' / 'client_medgemma_patients_mcp.py'
tree = ast.parse(source.read_text(encoding='utf-8-sig'))
names = {'update_connection_address', 'refresh_models', 'send_question', 'cancel_request'}
module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value
class Button:
    def __init__(self): self.options = {'text':'Connect'}
    def config(self, **options): self.options.update(options)
    configure = config
    def cget(self, key): return self.options[key]
class Worker:
    def __init__(self, target, **kwargs): self.target = target
    def start(self): self.target()
statuses = []
readiness = []
calls = []
finished = []
ns = dict(tk=SimpleNamespace(NORMAL='normal',DISABLED='disabled'),
          threading=SimpleNamespace(Thread=Worker), server_ip=Value('192.0.2.10'),
          server_url=Value(''), active_server_url=None, refresh_button=Button(),
          server_entry=Button(), model_picker=Button(), model_name=Value('example'),
          DEFAULT_MODELS=('example',), current_request=None,
          base_url=lambda value: value, show_status=statuses.append,
          set_readiness=lambda *args: readiness.append(args),
          post_ui=lambda callback: callback(),
          list_models=lambda url: calls.append(url) or ['example'],
          server_error_category=lambda error: str(error),
          finish_request=lambda *args, **kwargs: finished.append((args,kwargs)))
exec(compile(module,str(source),'exec'),ns)
ns['send_question']()
assert 'Connect to' in statuses[-1] and not calls
ns['refresh_models']()
assert ns['active_server_url'] == 'http://192.0.2.10:11434'
assert ns['refresh_button'].cget('text') == 'Disconnect'
closed = []
state = SimpleNamespace(terminal=False,cancel_event=threading.Event(),connection_lock=threading.Lock(),
                        response=SimpleNamespace(close=lambda:closed.append(True)),
                        snapshot=SimpleNamespace(request_id='synthetic-request'))
ns['current_request'] = state
ns['refresh_models']()
assert state.cancel_event.is_set() and closed and finished
assert ns['active_server_url'] is None and ns['server_ip'].get() == '' and ns['server_url'].get() == ''
assert ns['refresh_button'].cget('text') == 'Connect' and readiness[-1][0] == 'OFFLINE'
assert len(calls) == 1, 'Disconnect must not contact localhost or the old server'
ns['send_question']()
assert 'Connect to' in statuses[-1] and len(calls) == 1
ns['current_request'] = None
ns['refresh_models']()
assert calls[-1] == 'http://localhost:11434' and ns['active_server_url'] == calls[-1]
assert ns['server_ip'].get() == 'localhost', 'Blank Connect must display the default host'
ns['refresh_models']()
def fail(url): raise RuntimeError('synthetic failure')
ns['list_models'] = fail
ns['refresh_models']()
assert ns['active_server_url'] is None and ns['refresh_button'].cget('text') == 'Fail'
print('PASS: connect, disconnect, cancel/close response, block disconnected requests, explicit localhost reconnect, failure')
compile(source.read_text(encoding='utf-8-sig'),str(source),'exec')
print('PASS: GUI syntax')
