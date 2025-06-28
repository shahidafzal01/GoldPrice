'use client';

import { useEffect, useState } from 'react';
import { Loader2, TrendingUp } from 'lucide-react';

async function Last60(): Promise<number[]> {
  const res = await fetch('http://localhost:3001/history');
  if (!res.ok) throw new Error('Failed to fetch from local gold history');
  const data = await res.json();
  return data.prices;
}

async function fetchUSDtoINR(): Promise<number> {
  const res = await fetch('https://api.frankfurter.app/latest?from=USD&to=INR');
  const data = await res.json();
  return data.rates.INR;
}

export default function PredictionPage() {
  const [usdPrediction, setUsdPrediction] = useState<number | null>(null);
  const [inrPerGram, setInrPerGram] = useState<number | null>(null);
  const [inrPer10Gram, setInrPer10Gram] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function predict() {
    try {
      setLoading(true);
      setError(null);
      const prices = await Last60();

      const res = await fetch('http://localhost:3001/predictN', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: prices, days: 1 }),
      });

      if (!res.ok) throw new Error('Failed to get prediction');

      const data = await res.json();
      const usd = data.predictions[0];
      const usdToInr = await fetchUSDtoINR();
      const inrGram = (usd * usdToInr) / 31.1035;

      setUsdPrediction(usd);
      setInrPerGram(inrGram);
      setInrPer10Gram(inrGram * 10);
    } catch (err: any) {
      setError(err.message);
      setUsdPrediction(null);
      setInrPerGram(null);
      setInrPer10Gram(null);
    } finally {
      setLoading(false);
    }
  }

  // Clear function to reset all states
  function clear() {
    setUsdPrediction(null);
    setInrPerGram(null);
    setInrPer10Gram(null);
    setError(null);
    setLoading(false);
  }

  useEffect(() => {
    predict();
  }, []);

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-yellow-50 to-yellow-100 p-6">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center space-y-4 border border-yellow-300">
        <div className="flex items-center justify-center gap-2 text-yellow-700">
          <TrendingUp className="w-6 h-6" />
          <h1 className="text-2xl font-semibold">Gold Price Prediction</h1>
        </div>

        {loading ? (
          <div className="text-yellow-600 flex justify-center items-center gap-2">
            <Loader2 className="animate-spin h-5 w-5" />
            <span>Loading prediction...</span>
          </div>
        ) : error ? (
          <p className="text-red-600 font-medium">❌ {error}</p>
        ) : usdPrediction !== null ? (
          <div className="text-black space-y-2">
            <p>
              <strong>💵 USD/oz:</strong> ${usdPrediction.toFixed(2)}
            </p>
            <p>
              <strong>🇮🇳 INR/g:</strong> ₹{inrPerGram?.toFixed(2)}
            </p>
            <p>
              <strong>🇮🇳 INR/10g:</strong> ₹{inrPer10Gram?.toFixed(2)}
            </p>
          </div>
        ) : (
          <p className="text-gray-600">No prediction yet.</p>
        )}

        <div className="flex justify-center gap-4 mt-4">
          <button
            onClick={predict}
            disabled={loading}
            className="px-6 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 disabled:opacity-50"
          >
            Predict Again
          </button>

          <button
            onClick={clear}
            disabled={loading}
            className="px-6 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </div>
    </main>
  );
}

