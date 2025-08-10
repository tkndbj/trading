let portfolioChart = null;
let performanceChart = null;
let portfolioHistory = []; // Track portfolio value over time

// Initialize dashboard
document.addEventListener("DOMContentLoaded", function () {
  initPortfolioChart();
  initPerformanceChart();
  fetchData();

  // Update every 15 seconds
  setInterval(fetchData, 15000);
});

function initPortfolioChart() {
  const ctx = document.getElementById("portfolio-chart").getContext("2d");
  portfolioChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Cash"],
      datasets: [
        {
          data: [100],
          backgroundColor: ["#60a5fa"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#9ca3af",
            padding: 15,
            font: {
              size: 11,
            },
          },
        },
      },
    },
  });
}

function initPerformanceChart() {
  const ctx = document.getElementById("performance-chart");
  if (!ctx) return;

  performanceChart = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Portfolio Value",
          data: [],
          borderColor: "#60a5fa",
          backgroundColor: "rgba(96, 165, 250, 0.1)",
          borderWidth: 2,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          grid: {
            color: "#374151",
            drawBorder: false,
          },
          ticks: {
            color: "#9ca3af",
            font: {
              size: 10,
            },
          },
        },
        y: {
          grid: {
            color: "#374151",
            drawBorder: false,
          },
          ticks: {
            color: "#9ca3af",
            font: {
              size: 10,
            },
            callback: function (value) {
              return "$" + value.toFixed(0);
            },
          },
        },
      },
    },
  });
}

async function fetchData() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    if (data.error) {
      console.error("API Error:", data.error);
      document.querySelector(".connection-status").innerHTML =
        '<span class="status-dot status-offline"></span><span class="text-red-400 font-medium">ERROR</span>';
      return;
    }

    updateSummaryCards(data);
    updateMarketData(data.market_data);
    updatePositions(data.positions, data.market_data);
    updateTradeHistory(data.trade_history);
    updatePortfolioChart(data);
    updatePerformanceChart(data);
    updateLearningMetrics(data.learning_metrics);
    updateCostTracking(data.cost_tracking);

    // Store for global use
    window.data = data;

    // Update connection status
    document.querySelector(".connection-status").innerHTML =
      '<span class="status-dot status-live pulse"></span><span class="text-green-400 font-medium">LIVE</span>';
  } catch (error) {
    console.error("Error fetching data:", error);
    document.querySelector(".connection-status").innerHTML =
      '<span class="status-dot status-offline"></span><span class="text-red-400 font-medium">OFFLINE</span>';
  }
}

function updateSummaryCards(data) {
  // Update total value (portfolio value from backend)
  document.getElementById(
    "total-value"
  ).textContent = `$${data.total_value.toFixed(2)}`;

  // Update balance (actual USDC/BNFCR balance)
  document.getElementById("balance").textContent = `$${data.balance.toFixed(
    2
  )}`;

  // Update positions count
  document.getElementById("active-positions").textContent = Object.keys(
    data.positions || {}
  ).length;

  // Update trades count from trade history length
  document.getElementById("total-trades").textContent = data.trade_history
    ? data.trade_history.length
    : 0;
}

function updateMarketData(marketData) {
  const container = document.getElementById("market-data");
  container.innerHTML = "";

  if (!marketData || Object.keys(marketData).length === 0) {
    container.innerHTML =
      '<p class="text-gray-400 text-center">Loading market data...</p>';
    return;
  }

  const sortedCoins = Object.entries(marketData).sort(
    (a, b) => b[1].change_24h - a[1].change_24h
  );

  sortedCoins.forEach(([coin, data]) => {
    const changeColor =
      data.change_24h >= 0 ? "text-green-400" : "text-red-400";
    const changeIcon = data.change_24h >= 0 ? "fa-arrow-up" : "fa-arrow-down";

    const volumeIndicator =
      data.quote_volume > 100000000
        ? "🔥"
        : data.quote_volume > 50000000
        ? "📊"
        : "💤";

    container.innerHTML += `
      <div class="flex items-center justify-between p-3 bg-gray-700/20 rounded-lg hover:bg-gray-700/40 transition-all duration-200 border border-gray-700/30">
        <div class="flex items-center">
          <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg flex items-center justify-center mr-3 shadow-lg">
            <span class="font-bold text-xs text-white">${coin.slice(
              0,
              2
            )}</span>
          </div>
          <div>
            <p class="font-semibold text-sm">${coin}/USDT</p>
            <p class="text-gray-400 text-xs">$${data.price.toLocaleString()}</p>
          </div>
        </div>
        <div class="text-right">
          <p class="${changeColor} font-semibold text-sm flex items-center justify-end">
            <i class="fas ${changeIcon} text-xs mr-1"></i>
            ${data.change_24h >= 0 ? "+" : ""}${data.change_24h.toFixed(2)}%
          </p>
          <p class="text-gray-400 text-xs">
            Vol: ${(data.volume / 1000000).toFixed(1)}M ${volumeIndicator}
          </p>
        </div>
      </div>
    `;
  });
}

function updatePositions(positions, marketData) {
  const tbody = document.getElementById("positions-body");

  if (!positions || Object.keys(positions).length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="text-center py-12 text-gray-400">
          <div class="flex flex-col items-center">
            <i class="fas fa-inbox text-3xl mb-3 opacity-50"></i>
            <span class="text-sm">No active positions</span>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = "";

  Object.entries(positions).forEach(([symbol, position]) => {
    const coin = position.coin;
    const currentPrice = marketData[coin]?.price || position.mark_price;
    const entryPrice = position.entry_price;
    const direction = position.direction;
    const leverage = position.leverage;
    const notional = position.notional || 0;
    const positionSize = Math.abs(position.size);

    // Calculate PnL percentage and dollar amount
    let pnlPercent = 0;
    let pnlDollar = 0;

    if (direction === "LONG") {
      pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100;
      // For LONG: PnL = (current_price - entry_price) * position_size
      pnlDollar = (currentPrice - entryPrice) * positionSize;
    } else {
      pnlPercent = ((entryPrice - currentPrice) / entryPrice) * 100;
      // For SHORT: PnL = (entry_price - current_price) * position_size
      pnlDollar = (entryPrice - currentPrice) * positionSize;
    }

    // Use backend PnL if available and non-zero, otherwise use calculated
    const displayPnl =
      position.pnl && position.pnl !== 0 ? position.pnl : pnlDollar;

    const pnlColor = displayPnl >= 0 ? "text-green-400" : "text-red-400";
    const directionColor =
      direction === "LONG" ? "text-green-400" : "text-red-400";
    const directionIcon =
      direction === "LONG" ? "fa-arrow-up" : "fa-arrow-down";

    tbody.innerHTML += `
      <tr class="border-b border-gray-700/30 hover:bg-gray-800/30 transition-colors">
        <td class="py-3 px-3 font-semibold">
          <span class="${directionColor} flex items-center">
            <i class="fas ${directionIcon} mr-1"></i>${direction}
          </span>
          <span class="text-white">${coin}</span>
        </td>
        <td class="py-3 px-3 font-mono text-sm">${positionSize.toFixed(4)}</td>
        <td class="py-3 px-3 font-mono text-sm">$${entryPrice.toLocaleString()}</td>
        <td class="py-3 px-3 font-mono text-sm">$${currentPrice.toLocaleString()}</td>
        <td class="py-3 px-3 ${pnlColor} font-semibold">
          $${displayPnl >= 0 ? "+" : ""}${displayPnl.toFixed(2)}
          <br><span class="text-sm">(${
            pnlPercent >= 0 ? "+" : ""
          }${pnlPercent.toFixed(1)}%)</span>
        </td>
        <td class="py-3 px-3 text-sm font-bold">${leverage}x</td>
        <td class="py-3 px-3 text-sm">$${notional.toLocaleString()}</td>
      </tr>
    `;
  });
}

function updateTradeHistory(tradeHistory) {
  const tbody = document.getElementById("trades-body");

  if (!tradeHistory || tradeHistory.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="text-center py-12 text-gray-400">
          <div class="flex flex-col items-center">
            <i class="fas fa-chart-line text-3xl mb-3 opacity-50"></i>
            <span class="text-sm">No trades yet - AI analyzing markets</span>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = "";

  // Show last 20 trades, most recent first
  tradeHistory
    .slice(-20)
    .reverse()
    .forEach((trade) => {
      const actionColor = trade.action?.includes("LONG")
        ? "text-green-400"
        : "text-red-400";
      const actionIcon = trade.action?.includes("LONG")
        ? "fa-arrow-up"
        : "fa-arrow-down";

      const tradeTime = new Date(trade.timestamp || trade.time);
      const timeString = tradeTime.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      tbody.innerHTML += `
      <tr class="border-b border-gray-700/30 hover:bg-gray-800/30 transition-colors">
        <td class="py-3 px-3 text-sm font-mono">${timeString}</td>
        <td class="py-3 px-3 font-semibold">${trade.coin}</td>
        <td class="py-3 px-3 ${actionColor}">
          <span class="flex items-center">
            <i class="fas ${actionIcon} mr-1"></i>${
        trade.action || trade.direction
      }
          </span>
        </td>
        <td class="py-3 px-3 font-mono text-sm">$${
          trade.price?.toLocaleString() || "-"
        }</td>
        <td class="py-3 px-3 text-sm">${
          trade.position_size ? `$${trade.position_size.toFixed(2)}` : "-"
        }</td>
        <td class="py-3 px-3 text-sm font-bold">${trade.leverage || "-"}x</td>
        <td class="py-3 px-3 text-center">
          <span class="text-xs font-semibold px-2 py-1 rounded-full ${
            (trade.confidence || 5) >= 8
              ? "bg-green-900/50 text-green-400 border border-green-500/30"
              : (trade.confidence || 5) >= 6
              ? "bg-yellow-900/50 text-yellow-400 border border-yellow-500/30"
              : "bg-red-900/50 text-red-400 border border-red-500/30"
          }">
            ${trade.confidence || 5}/10
          </span>
        </td>
      </tr>
    `;
    });
}

function updateLearningMetrics(metrics) {
  let metricsSection = document.getElementById("learning-metrics-section");
  if (!metricsSection) {
    const mainContainer = document.querySelector(".container");
    const metricsHTML = `
      <div class="collapsible-content hidden" id="learning-metrics-section" style="display: none;">
        <div class="compact-card rounded-xl p-5 hover-lift">
          <h2 class="text-lg font-bold mb-4 text-blue-400 flex items-center">
            <i class="fas fa-brain mr-2"></i>AI Learning Metrics
          </h2>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-3" id="learning-metrics">
          </div>
        </div>
      </div>
    `;
    mainContainer.insertAdjacentHTML("beforeend", metricsHTML);
  }

  const metricsContainer = document.getElementById("learning-metrics");
  if (metrics && metricsContainer) {
    metricsContainer.innerHTML = `
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold ${
          (metrics.win_rate || 0) >= 50 ? "text-green-400" : "text-red-400"
        }">${(metrics.win_rate || 0).toFixed(1)}%</p>
        <p class="text-gray-400 text-xs">Win Rate</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-yellow-400">${
          metrics.best_leverage || 15
        }x</p>
        <p class="text-gray-400 text-xs">Best Leverage</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-blue-400">${
          metrics.total_trades || 0
        }</p>
        <p class="text-gray-400 text-xs">Total Trades</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-purple-400">
          ${(
            (metrics.avg_profit || 0) - Math.abs(metrics.avg_loss || 0)
          ).toFixed(2)}
        </p>
        <p class="text-gray-400 text-xs">Avg P&L</p>
      </div>
    `;
  }
}

function updateCostTracking(costData) {
  if (!costData) return;

  let costSection = document.getElementById("cost-tracking-section");
  if (!costSection) {
    const mainContainer = document.querySelector(".container");
    const costHTML = `
      <div class="collapsible-content hidden" id="cost-tracking-section" style="display: none;">
        <div class="compact-card rounded-xl p-5 hover-lift">
          <h2 class="text-lg font-bold mb-4 text-yellow-400 flex items-center">
            <i class="fas fa-dollar-sign mr-2"></i>Service Costs & Projections
          </h2>
          
          <div class="mb-6">
            <h3 class="text-base font-semibold text-gray-300 mb-3">Current Session</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3" id="current-costs">
            </div>
          </div>
          
          <div>
            <h3 class="text-base font-semibold text-gray-300 mb-3">Projected Costs</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3" id="cost-projections">
            </div>
          </div>
        </div>
      </div>
    `;

    mainContainer.insertAdjacentHTML("beforeend", costHTML);
  }

  const currentCosts = document.getElementById("current-costs");
  if (currentCosts) {
    currentCosts.innerHTML = `
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-blue-400">$${
          costData.current?.openai?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-xs">OpenAI</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-purple-400">$${
          costData.current?.railway?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-xs">Railway</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-yellow-400">$${
          costData.current?.total?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-xs">Total</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-cyan-400">${
          costData.current?.api_calls || 0
        }</p>
        <p class="text-gray-400 text-xs">API Calls</p>
      </div>
    `;
  }

  const projections = document.getElementById("cost-projections");
  if (projections && costData.projections) {
    projections.innerHTML = `
      <div class="bg-gradient-to-br from-blue-900/50 to-blue-800/50 rounded-lg p-4 border border-blue-500/20">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Weekly Estimate</h4>
        <p class="text-xl font-bold text-white">$${
          costData.projections.weekly?.total?.toFixed(2) || "0.00"
        }</p>
        <p class="text-xs text-gray-300">OpenAI: $${
          costData.projections.weekly?.openai?.toFixed(2) || "0.00"
        }</p>
        <p class="text-xs text-gray-300">Railway: $${
          costData.projections.weekly?.railway?.toFixed(2) || "0.00"
        }</p>
      </div>
      
      <div class="bg-gradient-to-br from-purple-900/50 to-purple-800/50 rounded-lg p-4 border border-purple-500/20">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Monthly Estimate</h4>
        <p class="text-xl font-bold text-white">$${
          costData.projections.monthly?.total?.toFixed(2) || "0.00"
        }</p>
        <p class="text-xs text-gray-300">OpenAI: $${
          costData.projections.monthly?.openai?.toFixed(2) || "0.00"
        }</p>
        <p class="text-xs text-gray-300">Railway: $${
          costData.projections.monthly?.railway?.toFixed(2) || "0.00"
        }</p>
      </div>
    `;
  }
}

function updatePortfolioChart(data) {
  if (!portfolioChart) return;

  const positions = Object.values(data.positions || {});
  const labels = ["Cash"];
  const values = [data.balance || 0];
  const colors = ["#60a5fa"];

  positions.forEach((position, index) => {
    labels.push(`${position.coin} ${position.direction}`);
    values.push(Math.abs(position.size * position.mark_price) || 0);
    colors.push(getColorForIndex(index));
  });

  portfolioChart.data.labels = labels;
  portfolioChart.data.datasets[0].data = values;
  portfolioChart.data.datasets[0].backgroundColor = colors;
  portfolioChart.update();
}

function updatePerformanceChart(data) {
  if (!performanceChart) return;

  const currentTime = new Date().toLocaleTimeString();
  const currentValue = data.total_value || 0;

  // Add to history
  portfolioHistory.push({
    time: currentTime,
    value: currentValue,
    timestamp: Date.now(),
  });

  // Keep only last 50 data points
  if (portfolioHistory.length > 50) {
    portfolioHistory = portfolioHistory.slice(-50);
  }

  // Update chart with historical data
  performanceChart.data.labels = portfolioHistory.map((h) => h.time);
  performanceChart.data.datasets[0].data = portfolioHistory.map((h) => h.value);
  performanceChart.update();
}

function getColorForIndex(index) {
  const colors = [
    "#f87171",
    "#fb923c",
    "#fbbf24",
    "#a3e635",
    "#34d399",
    "#22d3ee",
    "#60a5fa",
    "#a78bfa",
    "#f472b6",
    "#fb7185",
  ];
  return colors[index % colors.length];
}
