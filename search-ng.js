
class searchng {
  constructor(config = {}) {
    this.config = {
      workerTimeout: 15000,
      maxRetries: 3,
      cacheEnabled: true,
      cacheTTL: 3600000,
      rateLimit: 100,
      rateLimitWindow: 60000,
      debugMode: false,
      ...config
    };
    
    this.worker = null;
    this.cache = new Map();
    this.requestLog = [];
    this._initWorker();
  }

  _initWorker() {
    const workerCode = `
      self.pHash = (img) => {
        const size = 32, reducedSize = 8;
        const grayscale = new Uint8ClampedArray(size * size);
        
        for (let y = 0; y < size; y++) {
          for (let x = 0; x < size; x++) {
            const sx = Math.floor(x * img.width / size);
            const sy = Math.floor(y * img.height / size);
            const idx = (sy * img.width + sx) * 4;
            grayscale[y * size + x] = (img.data[idx] + img.data[idx + 1] + img.data[idx + 2]) / 3;
          }
        }
        
        const dct = new Float32Array(reducedSize * reducedSize);
        for (let i = 0; i < reducedSize; i++) {
          for (let j = 0; j < reducedSize; j++) {
            let sum = 0;
            for (let x = 0; x < size; x++) {
              for (let y = 0; y < size; y++) {
                sum += grayscale[y * size + x] * 
                       Math.cos((2 * x + 1) * i * Math.PI / (2 * size)) * 
                       Math.cos((2 * y + 1) * j * Math.PI / (2 * size));
              }
            }
            dct[i * reducedSize + j] = sum;
          }
        }
        
        const avg = dct.slice(1).reduce((a, b) => a + b) / (dct.length - 1);
        return dct.map(v => v > avg ? '1' : '0').join('');
      };

      self.hammingDistance = (h1, h2) => {
        if (h1.length !== h2.length) return Infinity;
        let dist = 0;
        for (let i = 0; i < h1.length; i++) {
          if (h1[i] !== h2[i]) dist++;
        }
        return dist;
      };

      self.levenshtein = (s1, s2, maxDist = Infinity) => {
        const len1 = s1.length, len2 = s2.length;
        
        if (Math.abs(len1 - len2) > maxDist) return maxDist + 1;
        
        let prev = Array(len2 + 1).fill(0).map((_, i) => i);
        let curr = Array(len2 + 1).fill(0);
        
        for (let i = 1; i <= len1; i++) {
          curr[0] = i;
          let minVal = i;
          
          for (let j = 1; j <= len2; j++) {
            const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
            curr[j] = Math.min(
              prev[j] + 1,
              curr[j - 1] + 1,
              prev[j - 1] + cost
            );
            minVal = Math.min(minVal, curr[j]);
          }
          
          if (minVal > maxDist) return maxDist + 1;
          [prev, curr] = [curr, prev];
        }
        
        return prev[len2];
      };

      self.fuzzySearch = (query, text, threshold = 0.3) => {
        const normalize = (str) => str.toLowerCase()
          .normalize('NFD').replace(/[\\u0300-\\u036f]/g, '')
          .trim();
        
        const q = normalize(query);
        const t = normalize(text);
        
        if (t.includes(q)) return 1.0;
        if (!q.length || !t.length) return 0;
        
        const maxLen = Math.max(q.length, t.length);
        const maxDist = Math.floor(maxLen * (1 - threshold));
        const dist = self.levenshtein(q, t, maxDist);
        
        if (dist > maxDist) return 0;
        return Math.max(0, 1 - dist / maxLen);
      };

      self.bm25Search = (query, documents, k1 = 1.2, b = 0.75) => {
        const normalize = (str) => str.toLowerCase()
          .normalize('NFD').replace(/[\\u0300-\\u036f]/g, '')
          .replace(/[^a-z0-9\\s]/g, '');
        
        const tokenize = (str) => {
          const words = normalize(str).split(/\\s+/).filter(Boolean);
          const tokens = new Set();
          
          words.forEach(word => {
            tokens.add(word);
            if (word.length > 3) {
              for (let i = 0; i <= word.length - 3; i++) {
                tokens.add(word.substring(i, i + 3));
              }
            }
          });
          
          return Array.from(tokens);
        };
        
        const terms = tokenize(query);
        const docCount = documents.length;
        
        const docTokens = documents.map(d => tokenize(d.text || d));
        const docLengths = docTokens.map(t => t.length);
        const avgDocLength = docLengths.reduce((a, b) => a + b, 0) / docCount;
        
        const idfCache = new Map();
        terms.forEach(term => {
          const df = docTokens.filter(tokens => tokens.includes(term)).length;
          idfCache.set(term, Math.log((docCount - df + 0.5) / (df + 0.5) + 1));
        });
        
        const scores = documents.map((doc, idx) => {
          const tokens = docTokens[idx];
          const docLength = docLengths[idx];
          let score = 0;
          
          terms.forEach(term => {
            const tf = tokens.filter(t => t === term).length;
            if (tf === 0) return;
            
            const idf = idfCache.get(term);
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / avgDocLength)));
          });
          
          const fullText = normalize(doc.text || doc);
          const fullQuery = normalize(query);
          if (fullText.includes(fullQuery)) {
            score *= 1.5;
          }
          
          return { idx, score };
        });
        
        return scores.sort((a, b) => b.score - a.score);
      };

      self.semanticBM25 = (query, documents, k1 = 1.2, b = 0.75) => {
        const normalize = (str) => str.toLowerCase()
          .normalize('NFD').replace(/[\\u0300-\\u036f]/g, '')
          .replace(/[^a-z0-9\\s]/g, '');
        
        const tokenize = (str) => {
          const words = normalize(str).split(/\\s+/).filter(Boolean);
          const tokens = new Set();
          
          words.forEach(word => {
            tokens.add(word);
            
            if (word.endsWith('ing')) tokens.add(word.slice(0, -3));
            if (word.endsWith('ed')) tokens.add(word.slice(0, -2));
            if (word.endsWith('s')) tokens.add(word.slice(0, -1));
            if (word.endsWith('ly')) tokens.add(word.slice(0, -2));
            
            if (word.length > 4) {
              for (let i = 0; i <= word.length - 4; i++) {
                tokens.add(word.substring(i, i + 4));
              }
            }
          });
          
          return Array.from(tokens);
        };
        
        const terms = tokenize(query);
        const docCount = documents.length;
        
        const docTokens = documents.map(d => tokenize(d.text || d));
        const docLengths = docTokens.map(t => t.length);
        const avgDocLength = docLengths.reduce((a, b) => a + b, 0) / docCount;
        
        const idfCache = new Map();
        terms.forEach(term => {
          const df = docTokens.filter(tokens => tokens.includes(term)).length;
          idfCache.set(term, Math.log((docCount - df + 0.5) / (df + 0.5) + 1));
        });
        
        const scores = documents.map((doc, idx) => {
          const tokens = docTokens[idx];
          const docLength = docLengths[idx];
          let score = 0;
          
          terms.forEach(term => {
            const tf = tokens.filter(t => t === term).length;
            if (tf === 0) return;
            
            const idf = idfCache.get(term);
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / avgDocLength)));
          });
          
          return { idx, score };
        });
        
        return scores.sort((a, b) => b.score - a.score);
      };

      self.audioFingerprint = (samples) => {
        const windowSize = 4096;
        const hopSize = 1024;
        const features = [];
        
        for (let i = 0; i < samples.length - windowSize; i += hopSize) {
          let energy = 0, zeroCrossings = 0, peak = 0;
          const window = samples.slice(i, i + windowSize);
          
          let spectralCentroid = 0, spectralFlux = 0;
          
          for (let j = 0; j < windowSize; j++) {
            const sample = window[j];
            energy += sample ** 2;
            if (Math.abs(sample) > peak) peak = Math.abs(sample);
            if (j > 0 && sample * window[j - 1] < 0) zeroCrossings++;
            spectralCentroid += Math.abs(sample) * j;
          }
          
          energy = Math.sqrt(energy / windowSize);
          const zcr = zeroCrossings / windowSize;
          spectralCentroid /= (energy * windowSize || 1);
          
          features.push(
            (energy > 0.1 ? '1' : '0') +
            (zcr > 0.05 ? '1' : '0') +
            (peak > 0.3 ? '1' : '0') +
            (spectralCentroid > 0.5 ? '1' : '0')
          );
        }
        
        return features.join('');
      };

      self.colorHistogram = (img) => {
        const bins = 32;
        const hist = new Array(bins * 3).fill(0);
        
        for (let i = 0; i < img.data.length; i += 4) {
          const r = Math.min(Math.floor(img.data[i] / 256 * bins), bins - 1);
          const g = Math.min(Math.floor(img.data[i + 1] / 256 * bins), bins - 1);
          const b = Math.min(Math.floor(img.data[i + 2] / 256 * bins), bins - 1);
          hist[r]++;
          hist[bins + g]++;
          hist[bins * 2 + b]++;
        }
        
        const total = img.width * img.height;
        return hist.map(v => v / total);
      };

      self.onmessage = (e) => {
        const { type, data, id } = e.data;
        const startTime = performance.now();
        
        try {
          let result;
          
          switch (type) {
            case 'phash':
              result = self.pHash(data);
              break;
            case 'hamming':
              result = self.hammingDistance(data.h1, data.h2);
              break;
            case 'fuzzy':
              result = self.fuzzySearch(data.query, data.text, data.threshold);
              break;
            case 'bm25':
              result = self.bm25Search(data.query, data.docs, data.k1, data.b);
              break;
            case 'semantic':
              result = self.semanticBM25(data.query, data.docs, data.k1, data.b);
              break;
            case 'audio':
              result = self.audioFingerprint(data);
              break;
            case 'color':
              result = self.colorHistogram(data);
              break;
            default:
              throw new Error('Unknown operation: ' + type);
          }
          
          self.postMessage({
            id,
            success: true,
            result,
            time: performance.now() - startTime
          });
        } catch (err) {
          self.postMessage({
            id,
            success: false,
            error: err.message
          });
        }
      };
    `;
    
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    this.worker = new Worker(URL.createObjectURL(blob));
  }

  _checkRateLimit() {
    const now = Date.now();
    this.requestLog = this.requestLog.filter(t => now - t < this.config.rateLimitWindow);
    
    if (this.requestLog.length >= this.config.rateLimit) {
      throw new Error('Rate limit exceeded');
    }
    
    this.requestLog.push(now);
  }

  _execute(type, data) {
    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36).substr(2, 9);
      const timeout = setTimeout(() => {
        reject(new Error('Worker timeout'));
      }, this.config.workerTimeout);
      
      const handler = (e) => {
        if (e.data.id === id) {
          clearTimeout(timeout);
          this.worker.removeEventListener('message', handler);
          
          if (e.data.success) {
            resolve(e.data);
          } else {
            reject(new Error(e.data.error));
          }
        }
      };
      
      this.worker.addEventListener('message', handler);
      this.worker.postMessage({ type, data, id });
    });
  }

  async fuzzySearch(query, items, options = {}) {
    this._checkRateLimit();
    
    const { threshold = 0.3, limit = 50 } = options;
    const cacheKey = `fuzzy_${query}_${threshold}`;
    
    if (this.config.cacheEnabled && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.config.cacheTTL) {
        return cached.data;
      }
    }
    
    const results = await Promise.all(
      items.map(async (item, idx) => {
        const text = typeof item === 'string' ? item : item.text;
        const res = await this._execute('fuzzy', { query, text, threshold });
        
        if (res.result > threshold) {
          return {
            ...(typeof item === 'object' ? item : { text: item }),
            score: res.result,
            index: idx
          };
        }
        return null;
      })
    );
    
    const filtered = results
      .filter(r => r !== null)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    
    if (this.config.cacheEnabled) {
      this.cache.set(cacheKey, { data: filtered, timestamp: Date.now() });
    }
    
    return filtered;
  }

  async bm25Search(query, documents, options = {}) {
    this._checkRateLimit();
    
    const { threshold = 0, limit = 50, k1 = 1.2, b = 0.75 } = options;
    const cacheKey = `bm25_${query}_${k1}_${b}`;
    
    if (this.config.cacheEnabled && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.config.cacheTTL) {
        return cached.data;
      }
    }
    
    const res = await this._execute('bm25', { query, docs: documents, k1, b });
    
    const filtered = res.result
      .filter(r => r.score > threshold)
      .slice(0, limit)
      .map(r => ({
        ...documents[r.idx],
        score: r.score,
        index: r.idx
      }));
    
    if (this.config.cacheEnabled) {
      this.cache.set(cacheKey, { data: filtered, timestamp: Date.now() });
    }
    
    return filtered;
  }

  async semanticSearch(query, documents, options = {}) {
    this._checkRateLimit();
    
    const { threshold = 0, limit = 50, k1 = 1.2, b = 0.75 } = options;
    
    const res = await this._execute('semantic', { query, docs: documents, k1, b });
    
    return res.result
      .filter(r => r.score > threshold)
      .slice(0, limit)
      .map(r => ({
        ...documents[r.idx],
        score: r.score,
        index: r.idx
      }));
  }

  async imageHash(imageData) {
    this._checkRateLimit();
    
    const cacheKey = `img_${imageData.width}x${imageData.height}_${imageData.data[0]}`;
    if (this.config.cacheEnabled && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }
    
    const res = await this._execute('phash', imageData);
    
    if (this.config.cacheEnabled) {
      this.cache.set(cacheKey, res.result);
    }
    
    return res.result;
  }

  async imageSimilarity(hash1, hash2) {
    const res = await this._execute('hamming', { h1: hash1, h2: hash2 });
    return 1 - (res.result / 64);
  }

  async findSimilarImages(targetHash, imageDB, options = {}) {
    this._checkRateLimit();
    
    const { threshold = 0.85, limit = 10 } = options;
    
    const results = await Promise.all(
      imageDB.map(async (img) => {
        const similarity = await this.imageSimilarity(targetHash, img.hash);
        return similarity >= threshold ? { ...img, similarity } : null;
      })
    );
    
    return results
      .filter(r => r !== null)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  async colorHistogram(imageData) {
    const res = await this._execute('color', imageData);
    return res.result;
  }

  findByColor(targetHist, imageDB, options = {}) {
    const { threshold = 0.8, limit = 10 } = options;
    
    const results = imageDB.map(img => {
      let intersection = 0;
      for (let i = 0; i < targetHist.length; i++) {
        intersection += Math.min(targetHist[i], img.histogram[i]);
      }
      
      return intersection >= threshold ? { ...img, colorMatch: intersection } : null;
    });
    
    return results
      .filter(r => r !== null)
      .sort((a, b) => b.colorMatch - a.colorMatch)
      .slice(0, limit);
  }

  async audioFingerprint(audioData) {
    this._checkRateLimit();
    const res = await this._execute('audio', audioData);
    return res.result;
  }

  findSimilarAudio(targetFP, audioDB, options = {}) {
    const { threshold = 0.7, limit = 10 } = options;
    
    const results = audioDB.map(audio => {
      let matches = 0;
      const len = Math.min(targetFP.length, audio.fingerprint.length);
      
      for (let i = 0; i < len; i++) {
        if (targetFP[i] === audio.fingerprint[i]) matches++;
      }
      
      const similarity = matches / len;
      return similarity >= threshold ? { ...audio, similarity } : null;
    });
    
    return results
      .filter(r => r !== null)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  getImageData(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.onload = (e) => {
        const img = new Image();
        
        img.onerror = () => reject(new Error('Failed to load image'));
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0);
          
          resolve(ctx.getImageData(0, 0, canvas.width, canvas.height));
        };
        
        img.src = e.target.result;
      };
      
      reader.readAsDataURL(file);
    });
  }

  async recordAudio(duration = 5000) {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 44100
      }
    });
    
    const recorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    const chunks = [];
    
    return new Promise((resolve, reject) => {
      recorder.ondataavailable = (e) => chunks.push(e.data);
      
      recorder.onstop = async () => {
        try {
          const blob = new Blob(chunks, { type: 'audio/webm' });
          const buffer = await blob.arrayBuffer();
          const ctx = new AudioContext({ sampleRate: 44100 });
          const audio = await ctx.decodeAudioData(buffer);
          
          stream.getTracks().forEach(track => track.stop());
          resolve(audio.getChannelData(0));
        } catch (err) {
          reject(err);
        }
      };
      
      recorder.onerror = (e) => reject(e.error);
      
      recorder.start();
      setTimeout(() => recorder.stop(), duration);
    });
  }

  clearCache() {
    this.cache.clear();
  }

  destroy() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.cache.clear();
    this.requestLog = [];
  }

  getStats() {
    return {
      cacheSize: this.cache.size,
      requestCount: this.requestLog.length,
      rateLimit: this.config.rateLimit
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = searchng;
}
if (typeof window !== 'undefined') {
  window.searchng = searchng;
}
