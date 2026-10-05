// Early access sign-ups for traceX: one JSON record per email in a private Vercel Blob store.
//
//   POST {email, role?, building?, from?}   join, or update your own record
//   POST {action: "remove", email}          remove yourself (always answers ok, so it can't be used to probe the list)
//   GET  Authorization: Bearer $SIGNUP_ADMIN_TOKEN   the list as CSV (?format=json for JSON)
//
// Records live under signups/<sha256 of the email>.json, so no address appears in a pathname or a listing.
import { createHash, timingSafeEqual } from 'node:crypto';
import { del, get, list, put } from '@vercel/blob';

const ROLES = new Set(['share', 'train', 'bounty', 'validate', '']);
const EMAIL = /^[^\s@<>"]{1,64}@[^\s@<>"]{1,189}\.[A-Za-z]{2,63}$/;
const HEADERS = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };

const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: HEADERS });
const clean = (value, max) => String(value ?? '').replace(/[\u0000-\u001f\u007f]/g, ' ').trim().slice(0, max);
const keyFor = email => `signups/${createHash('sha256').update(email).digest('hex').slice(0, 40)}.json`;

async function read(pathname) {
  try {
    const found = await get(pathname, { access: 'private', useCache: false });
    if (found?.statusCode !== 200) return null;
    return JSON.parse(await new Response(found.stream).text());
  } catch {
    return null;
  }
}

export async function POST(request) {
  let body;
  try {
    const text = await request.text();
    if (text.length > 4096) return reply(413, { ok: false, error: 'too_large' });
    body = JSON.parse(text);
  } catch {
    return reply(400, { ok: false, error: 'bad_request' });
  }
  if (!body || typeof body !== 'object') return reply(400, { ok: false, error: 'bad_request' });
  if (clean(body.website, 200)) return reply(200, { ok: true });  // honeypot: only bots fill the hidden field

  const email = clean(body.email, 254).toLowerCase();
  if (!EMAIL.test(email)) return reply(400, { ok: false, error: 'email' });
  const pathname = keyFor(email);

  if (body.action === 'remove') {
    try { await del(pathname); } catch { /* nothing stored for that address */ }
    return reply(200, { ok: true, removed: true });
  }

  const before = await read(pathname);
  const now = new Date().toISOString();
  const role = ROLES.has(body.role) ? body.role : '';
  const building = clean(body.building, 280);
  const record = {
    email,
    role: role || before?.role || '',
    building: building || before?.building || '',
    from: clean(body.from, 20) || before?.from || '',
    created_at: before?.created_at || now,
    updated_at: now,
  };
  await put(pathname, JSON.stringify(record), {
    access: 'private',
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: 'application/json',
  });
  return reply(200, { ok: true, returning: Boolean(before) });
}

function authorized(request) {
  const want = process.env.SIGNUP_ADMIN_TOKEN || '';
  const got = (request.headers.get('authorization') || '').replace(/^Bearer\s+/i, '');
  if (want.length < 24 || got.length !== want.length) return false;
  return timingSafeEqual(Buffer.from(got), Buffer.from(want));
}

// Spreadsheets run cells that start with = + - @ as formulas; a leading quote keeps them as text.
const cell = value => {
  const text = String(value ?? '');
  const safe = /^[=+\-@]/.test(text) ? `'${text}` : text;
  return `"${safe.replace(/"/g, '""')}"`;
};

export async function GET(request) {
  if (!authorized(request)) return reply(401, { ok: false });
  const rows = [];
  let cursor;
  do {
    const page = await list({ prefix: 'signups/', cursor, limit: 1000 });
    for (const blob of page.blobs) {
      const record = await read(blob.pathname);
      if (record) rows.push(record);
    }
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  rows.sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));

  if (new URL(request.url).searchParams.get('format') === 'json') {
    return reply(200, { ok: true, count: rows.length, signups: rows });
  }
  const fields = ['email', 'role', 'building', 'from', 'created_at', 'updated_at'];
  const csv = [fields.join(','), ...rows.map(r => fields.map(f => cell(r[f])).join(','))].join('\r\n');
  return new Response(csv, {
    headers: {
      'Content-Type': 'text/csv; charset=utf-8',
      'Content-Disposition': 'attachment; filename="tracex-signups.csv"',
      'Cache-Control': 'no-store',
    },
  });
}
