const CACHE_NAME = 'ranofocus-4-v1.0.0'
const STATIC_CACHE_URLS = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png'
]

const API_CACHE_URLS = [
  '/api/geophysics/sample-data'
]

// Install event - cache static resources
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...')
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker: Caching static files')
        return cache.addAll(STATIC_CACHE_URLS)
      })
      .catch((error) => {
        console.error('Service Worker: Cache installation failed', error)
      })
  )
  self.skipWaiting()
})

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...')
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Deleting old cache', cacheName)
            return caches.delete(cacheName)
          }
        })
      )
    })
  )
  self.clients.claim()
})

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      handleApiRequest(request)
    )
    return
  }

  // Handle static resources
  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          console.log('Service Worker: Serving from cache', request.url)
          return cachedResponse
        }

        console.log('Service Worker: Fetching from network', request.url)
        return fetch(request)
          .then((response) => {
            // Don't cache non-successful responses
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response
            }

            // Clone the response for caching
            const responseToCache = response.clone()
            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(request, responseToCache)
              })

            return response
          })
          .catch((error) => {
            console.error('Service Worker: Fetch failed', error)
            
            // Return offline page for navigation requests
            if (request.destination === 'document') {
              return caches.match('/')
            }
            
            throw error
          })
      })
  )
})

// Handle API requests with network-first strategy
async function handleApiRequest(request) {
  const url = new URL(request.url)
  
  try {
    // Try network first
    const networkResponse = await fetch(request)
    
    // Cache successful GET requests
    if (request.method === 'GET' && networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME)
      cache.put(request, networkResponse.clone())
    }
    
    return networkResponse
  } catch (error) {
    console.log('Service Worker: Network failed, trying cache for API request')
    
    // Fallback to cache for GET requests
    if (request.method === 'GET') {
      const cachedResponse = await caches.match(request)
      if (cachedResponse) {
        return cachedResponse
      }
    }
    
    // Return offline response for failed API requests
    return new Response(
      JSON.stringify({
        success: false,
        error: 'Application hors ligne - Fonctionnalité limitée',
        offline: true
      }),
      {
        status: 503,
        statusText: 'Service Unavailable',
        headers: {
          'Content-Type': 'application/json'
        }
      }
    )
  }
}

// Background sync for data processing
self.addEventListener('sync', (event) => {
  console.log('Service Worker: Background sync triggered', event.tag)
  
  if (event.tag === 'process-geophysics-data') {
    event.waitUntil(
      processQueuedData()
    )
  }
})

// Process queued geophysics data when back online
async function processQueuedData() {
  try {
    // Get queued data from IndexedDB or localStorage
    const queuedRequests = await getQueuedRequests()
    
    for (const request of queuedRequests) {
      try {
        const response = await fetch(request.url, {
          method: request.method,
          headers: request.headers,
          body: request.body
        })
        
        if (response.ok) {
          // Remove from queue and notify client
          await removeFromQueue(request.id)
          await notifyClient('data-processed', { requestId: request.id, success: true })
        }
      } catch (error) {
        console.error('Service Worker: Failed to process queued request', error)
      }
    }
  } catch (error) {
    console.error('Service Worker: Background sync failed', error)
  }
}

// Notify clients of background sync results
async function notifyClient(type, data) {
  const clients = await self.clients.matchAll()
  clients.forEach(client => {
    client.postMessage({ type, data })
  })
}

// Placeholder functions for queue management
async function getQueuedRequests() {
  // In a real implementation, this would read from IndexedDB
  return []
}

async function removeFromQueue(requestId) {
  // In a real implementation, this would remove from IndexedDB
  console.log('Service Worker: Removing request from queue', requestId)
}

// Handle push notifications (for future use)
self.addEventListener('push', (event) => {
  console.log('Service Worker: Push notification received')
  
  const options = {
    body: 'Analyse géophysique terminée',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    vibrate: [200, 100, 200],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'Voir les résultats',
        icon: '/icon-192.png'
      },
      {
        action: 'close',
        title: 'Fermer',
        icon: '/icon-192.png'
      }
    ]
  }
  
  event.waitUntil(
    self.registration.showNotification('RanoFocus 4', options)
  )
})

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('Service Worker: Notification clicked', event.action)
  
  event.notification.close()
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    )
  }
})

console.log('Service Worker: Loaded successfully')

