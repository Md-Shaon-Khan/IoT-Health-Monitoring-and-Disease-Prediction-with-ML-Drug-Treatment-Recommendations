// Service Worker for PWA
const CACHE_NAME = 'healthbridge-v1.0.0';
const urlsToCache = [
  '/',
  '/static/style.css',
  '/static/dashboard.css',
  '/static/auth.css',
  '/static/service-style.css',
  '/static/script.js',
  '/static/dashboard.js',
  '/static/auth.js',
  '/static/predictor.js',
  '/static/navbar.js',
  '/static/manifest.json'
];

// Install Service Worker
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
      })
  );
});

// Fetch Event
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      })
  );
});

// Activate Event
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});