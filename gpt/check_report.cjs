// Run: node gpt/check_report.cjs (from the CP1 directory).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const html = fs.readFileSync(path.join(__dirname, 'domain-knowledge.html'), 'utf8');
const data = JSON.parse(html.match(/<script id="target-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Parse the complete browser program too.
const context = vm.createContext({Math, Array});
vm.runInContext(script.split('// UI initialization.')[0], context);
const {fourierCoefficients, reconstruct, circleGeometry, topicMatches} = context;
const close = (a, b, tolerance = 1e-9) => assert(Math.abs(a - b) < tolerance, `${a} != ${b}`);

// A translated ellipse contains only DC and first harmonics.
const ellipse = Array.from({length: 32}, (_, i) => [3 + 2 * Math.cos(2 * Math.PI * i / 32), -1 + Math.sin(2 * Math.PI * i / 32)]);
reconstruct(fourierCoefficients(ellipse), 1).forEach((p, i) => p.forEach((v, j) => close(v, ellipse[i][j])));
assert.equal(data.curves.length, 3);
data.curves.forEach(curve => {
  assert.equal(curve.length, 200);
  assert(curve.every(p => p.length === 2 && p.every(Number.isFinite)));
  const coeff = fourierCoefficients(curve);
  reconstruct(coeff, 100).forEach((p, i) => p.forEach((v, j) => close(v, curve[i][j])));
  const error = k => reconstruct(coeff, k).reduce((s, p, i) => s + (p[0] - curve[i][0]) ** 2 + (p[1] - curve[i][1]) ** 2, 0);
  assert(error(30) < error(3), 'Adding harmonics should reduce reconstruction error');
});

// Closure includes two branches, tangency, and both kinds of nonintersection.
const regular = circleGeometry(1, .8, 1.2);
assert(regular.valid && regular.h > 0);
close(Math.hypot(regular.x, regular.h), 1);
close(Math.hypot(regular.x - 1.2, regular.h), .8);
close(Math.hypot(regular.x, -regular.h), 1);
for (const d of [.2, 1.8]) {const g = circleGeometry(1, .8, d); assert(g.valid); close(g.h, 0, 1e-7);}
for (const d of [.1, 1.9, 0]) assert.equal(circleGeometry(1, .8, d).valid, false);
assert(topicMatches('Geometry and SCALE', 'core', ' scale ', 'core'));
assert(!topicMatches('Geometry', 'core', '', 'next'));
assert(!topicMatches('Geometry', 'core', 'nonexistent', 'all'));

const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(ids).size, ids.length, 'HTML IDs must be unique');
for (const [, href] of html.matchAll(/href="([^"]+)"/g)) {
  if (href.startsWith('#')) assert(ids.includes(href.slice(1)), `Missing anchor: ${href}`);
  if (href.startsWith('../')) assert(fs.existsSync(path.resolve(__dirname, decodeURIComponent(href))), `Missing local file: ${href}`);
}
assert.equal((html.match(/class="topic" /g) || []).length, 10);
assert.equal((html.match(/class="source" /g) || []).length, 15);
assert(!html.includes('__TARGET_DATA__'));
const sourcePath = path.resolve(__dirname, '..', data.source);
if (fs.existsSync(sourcePath)) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(sourcePath)).digest('hex'), data.sha256);
console.log('PASS: Fourier reconstruction, closure boundaries, topic filters, provenance, all anchors and local links, and JavaScript syntax.');
