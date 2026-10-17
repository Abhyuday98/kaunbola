// Service worker: makes the app installable, keeps the app shell available when the tunnel is down
// (so the phone shows our "retrying" banner instead of the browser's error page), and handles "share to Turnwise".
const SHELL = 'shell-v4';
self.addEventListener('install', e => e.waitUntil(caches.open(SHELL).then(c => c.addAll(['/', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png', '/fonts/baloo2.css'])).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method === 'GET' && url.origin === location.origin && !url.pathname.startsWith('/api/')) {
    // network first, cached copy if the network is down; a navigation always falls back to the shell
    e.respondWith(fetch(e.request).then(r => { if (r.ok) caches.open(SHELL).then(c => c.put(e.request, r.clone())); return r; })
      .catch(() => caches.match(e.request, {ignoreSearch: true}).then(r => r || (e.request.mode === 'navigate' ? caches.match('/') : Response.error()))));
    return;
  }
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
