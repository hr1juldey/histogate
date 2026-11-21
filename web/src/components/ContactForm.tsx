import { useState } from 'react';
import { apiBase } from '../lib/api';

/** Contact island: message + optional résumé attachment, posted to FastAPI. */
export default function ContactForm() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [msg, setMsg] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setState('sending');
    const fd = new FormData();
    fd.append('name', name);
    fd.append('email', email);
    fd.append('message', msg);
    if (file) fd.append('resume', file);
    try {
      const r = await fetch(`${apiBase}/contact`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(String(r.status));
      setState('sent');
    } catch { setState('error'); }
  };

  if (state === 'sent') {
    return (
      <div className="card sent">
        <h3>Received — thank you.</h3>
        <p>You'll hear back within a day. The full build log and measured artifacts
        are on GitHub in the meantime.</p>
      </div>
    );
  }

  return (
    <form className="card form" onSubmit={submit}>
      <div className="frow">
        <label>Name<input required value={name} onChange={e => setName(e.target.value)} /></label>
        <label>Email<input required type="email" value={email} onChange={e => setEmail(e.target.value)} /></label>
      </div>
      <label>Message<textarea rows={5} value={msg} onChange={e => setMsg(e.target.value)}
        placeholder="Working on digital pathology? Ask for a walkthrough, the build log, or the résumé behind this project." /></label>
      <label className="file">
        <span className="btn ghost">{file ? `📎 ${file.name}` : 'Attach résumé (PDF)'}</span>
        <input type="file" accept=".pdf,.doc,.docx" hidden
          onChange={e => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <div className="actions">
        <button className="btn primary" disabled={state === 'sending'}>
          {state === 'sending' ? 'Sending…' : 'Send message'}
        </button>
        {state === 'error' && <span className="err">backend unreachable — try again shortly</span>}
      </div>
    </form>
  );
}
