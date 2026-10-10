// Minimal service worker: makes the app installable and handles "share to Kaunbola" from other apps.
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method === 'POST' && url.pathname === '/share') {
    e.respondWith((async () => {
      const form = await e.request.formData();
      const files = form.getAll('audio').filter(f => f && f.size);
      let last = null, dup = false;
      for (const f of files) {
        const q = new URLSearchParams({filename: f.name || 'shared.m4a', created: String((f.lastModified || 0) / 1000)});
        const r = await fetch('/api/upload?' + q, {method: 'POST', body: f});
        if (r.ok) { const j = await r.json(); last = j.id; dup = dup || !!j.duplicate; }
      }
      return Response.redirect(files.length === 1 && last ? `/?shared=${dup ? 'dup' : 'ok'}#/note/${last}` : `/?shared=${files.length}#/`, 303);
    })());
  }
});
