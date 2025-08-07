let portfolioChart = null;
let performanceChart = null;

// Initialize dashboard
document.addEventListener("DOMContentLoaded", function () {
  initPortfolioChart();
  initPerformanceChart();
  fetchData();

  // Update every 15 seconds (more frequent for better real-time feel)
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
          data: [1000],
          backgroundColor: ["#60a5fa"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#9ca3af",
            padding: 20,
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
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          grid: {
            color: "#374151",
          },
          ticks: {
            color: "#9ca3af",
          },
        },
        y: {
          grid: {
            color: "#374151",
          },
          ticks: {
            color: "#9ca3af",
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

    updateSummaryCards(data);
    updateMarketData(data.market_data);
    updateLeveragePositions(data.positions, data.market_data);
    updateLeverageTradeHistory(data.trade_history);
    updatePortfolioChart(data);
    updatePerformanceChart(data);
    updateLearningMetrics(data.learning_metrics);
    updateCostTracking(data.cost_tracking);
    updateMarketRegimes(data.positions, data.market_data);

    // Store for global use
    window.data = data;

    // Update connection status
    document.querySelector(".connection-status").innerHTML =
      '<i class="fas fa-circle text-xs mr-1 text-green-400"></i><span class="text-sm text-green-400">LIVE</span>';
  } catch (error) {
    console.error("Error fetching data:", error);
    document.querySelector(".connection-status").innerHTML =
      '<i class="fas fa-circle text-xs mr-1 text-red-400"></i><span class="text-sm text-red-400">OFFLINE</span>';
  }
}

function updateSummaryCards(data) {
  document.getElementById(
    "total-value"
  ).textContent = `$${data.total_value.toFixed(2)}`;

  const pnlElement = document.getElementById("total-pnl");
  const pnl = data.total_value - 1000;
  const pnlPercent = (pnl / 1000) * 100;
  pnlElement.innerHTML = `
    $${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}
    <span class="text-sm font-normal">(${
      pnlPercent >= 0 ? "+" : ""
    }${pnlPercent.toFixed(1)}%)</span>
  `;
  pnlElement.className = `text-2xl font-bold ${
    pnl >= 0 ? "text-green-400" : "text-red-400"
  }`;

  document.getElementById("active-positions").textContent = Object.keys(
    data.positions
  ).length;
  document.getElementById("total-trades").textContent =
    data.trade_history.length;
}

function updateMarketRegimes(positions, marketData) {
  // Create market regimes section if not exists
  let regimeContainer = document.getElementById("market-regimes");
  if (!regimeContainer) {
    const marketSection = document.getElementById("market-data-section");
    if (!marketSection) return;

    const regimeHTML = `
      <div class="mt-4">
        <h3 class="text-sm font-semibold text-gray-400 mb-2">Market Regimes</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2" id="market-regimes">
          <!-- Regimes will be populated here -->
        </div>
      </div>
    `;
    marketSection.insertAdjacentHTML("beforeend", regimeHTML);
    regimeContainer = document.getElementById("market-regimes");
  }

  // Count positions by regime
  const regimeCounts = {};
  Object.values(positions).forEach((pos) => {
    const regime = pos.market_regime || "UNKNOWN";
    regimeCounts[regime] = (regimeCounts[regime] || 0) + 1;
  });

  const regimeIcons = {
    TRENDING_UP: "🚀",
    TRENDING_DOWN: "📉",
    RANGING: "↔️",
    VOLATILE: "⚡",
    TRANSITIONAL: "🔄",
    UNKNOWN: "❓",
  };

  const regimeColors = {
    TRENDING_UP: "text-green-400",
    TRENDING_DOWN: "text-red-400",
    RANGING: "text-blue-400",
    VOLATILE: "text-yellow-400",
    TRANSITIONAL: "text-purple-400",
    UNKNOWN: "text-gray-400",
  };

  let regimeHTML = "";
  for (const [regime, count] of Object.entries(regimeCounts)) {
    regimeHTML += `
      <div class="bg-gray-700 rounded p-2 text-center">
        <span class="text-lg">${regimeIcons[regime] || "?"}</span>
        <p class="text-xs ${
          regimeColors[regime] || "text-gray-400"
        }">${regime.replace("_", " ")}</p>
        <p class="text-sm font-bold">${count}</p>
      </div>
    `;
  }

  if (regimeHTML) {
    regimeContainer.innerHTML = regimeHTML;
  }
}

function updateCostTracking(costData) {
  if (!costData) return;

  // Create cost tracking section if not exists
  let costContainer = document.getElementById("cost-tracking");
  if (!costContainer) {
    const mainContainer = document.querySelector(".container");
    const costHTML = `
      <div class="mt-8" id="cost-tracking-section">
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 shadow-xl">
          <h2 class="text-xl font-bold mb-6 text-yellow-400">
            <i class="fas fa-dollar-sign mr-2"></i>Service Costs & Projections
          </h2>
          
          <!-- Current Costs -->
          <div class="mb-6">
            <h3 class="text-lg font-semibold text-gray-300 mb-3">Current Session</h3>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="current-costs">
              <!-- Current costs here -->
            </div>
          </div>
          
          <!-- Cost Projections -->
          <div>
            <h3 class="text-lg font-semibold text-gray-300 mb-3">Projected Costs</h3>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4" id="cost-projections">
              <!-- Projections here -->
            </div>
          </div>
        </div>
      </div>
    `;

    const learningSection = document.getElementById("learning-metrics-section");
    if (learningSection) {
      learningSection.insertAdjacentHTML("afterend", costHTML);
    } else {
      mainContainer.insertAdjacentHTML("beforeend", costHTML);
    }
  }

  // Update current costs
  const currentCosts = document.getElementById("current-costs");
  if (currentCosts) {
    const totalValue = window.data?.total_value || 1000;
    const netProfit = totalValue - 1000 - costData.current.total;

    currentCosts.innerHTML = `
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-blue-400">$${
          costData.current.openai?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-sm">OpenAI</p>
        <p class="text-xs text-gray-500">${
          costData.current.api_calls || 0
        } calls</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-purple-400">$${
          costData.current.railway?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-sm">Railway</p>
        <p class="text-xs text-gray-500">Hosting</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-yellow-400">$${
          costData.current.total?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-sm">Total Cost</p>
        <p class="text-xs text-gray-500">All services</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold ${
          netProfit >= 0 ? "text-green-400" : "text-red-400"
        }">
          $${netProfit.toFixed(2)}
        </p>
        <p class="text-gray-400 text-sm">Net Profit</p>
        <p class="text-xs text-gray-500">After costs</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-cyan-400">
          ${
            costData.current.total > 0
              ? ((totalValue - 1000) / costData.current.total).toFixed(1)
              : "∞"
          }x
        </p>
        <p class="text-gray-400 text-sm">ROI</p>
        <p class="text-xs text-gray-500">Return on cost</p>
      </div>
    `;
  }

  // Update projections
  const projections = document.getElementById("cost-projections");
  if (projections && costData.projections) {
    projections.innerHTML = `
      <div class="bg-gradient-to-br from-blue-900 to-blue-800 rounded-lg p-4">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Weekly Estimate</h4>
        <div class="space-y-1">
          <p class="text-xl font-bold text-white">$${
            costData.projections.weekly.total?.toFixed(2) || "0.00"
          }</p>
          <p class="text-xs text-gray-300">OpenAI: $${
            costData.projections.weekly.openai?.toFixed(2) || "0.00"
          }</p>
          <p class="text-xs text-gray-300">Railway: $${
            costData.projections.weekly.railway?.toFixed(2) || "0.00"
          }</p>
        </div>
      </div>
      
      <div class="bg-gradient-to-br from-purple-900 to-purple-800 rounded-lg p-4">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Monthly Estimate</h4>
        <div class="space-y-1">
          <p class="text-xl font-bold text-white">$${
            costData.projections.monthly.total?.toFixed(2) || "0.00"
          }</p>
          <p class="text-xs text-gray-300">OpenAI: $${
            costData.projections.monthly.openai?.toFixed(2) || "0.00"
          }</p>
          <p class="text-xs text-gray-300">Railway: $${
            costData.projections.monthly.railway?.toFixed(2) || "0.00"
          }</p>
        </div>
      </div>
      
      <div class="bg-gradient-to-br from-green-900 to-green-800 rounded-lg p-4">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Cost Efficiency</h4>
        <div class="space-y-1">
          <p class="text-xl font-bold text-white">
            $${(
              costData.current.total / (costData.current.api_calls || 1)
            ).toFixed(4)}
          </p>
          <p class="text-xs text-gray-300">Per API Call</p>
          <p class="text-xs text-gray-300 mt-2">
            ${
              costData.current.api_calls
                ? (costData.current.api_calls / 24).toFixed(1)
                : "0"
            } calls/hour avg
          </p>
        </div>
      </div>
    `;
  }
}

function updateMarketData(marketData) {
  const container = document.getElementById("market-data");
  container.innerHTML = "";

  // Sort by 24h change for better visibility
  const sortedCoins = Object.entries(marketData).sort(
    (a, b) => b[1].change_24h - a[1].change_24h
  );

  sortedCoins.forEach(([coin, data]) => {
    const changeColor =
      data.change_24h >= 0 ? "text-green-400" : "text-red-400";
    const changeIcon = data.change_24h >= 0 ? "fa-arrow-up" : "fa-arrow-down";

    // Volume indicator based on quote volume
    const volumeIndicator =
      data.quote_volume > 100000000
        ? "🔥"
        : data.quote_volume > 50000000
        ? "📊"
        : "💤";

    container.innerHTML += `
      <div class="flex items-center justify-between p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all duration-200">
        <div class="flex items-center">
          <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center mr-3 shadow-lg">
            <span class="font-bold text-xs text-white">${coin}</span>
          </div>
          <div>
            <p class="font-semibold text-sm">${coin}/USDT</p>
            <p class="text-gray-400 text-xs">$${data.price.toLocaleString()}</p>
          </div>
        </div>
        <div class="text-right">
          <p class="${changeColor} font-semibold text-sm">
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

function updateLeveragePositions(positions, marketData) {
  const tbody = document.getElementById("positions-body");

  if (Object.keys(positions).length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="text-center py-8 text-gray-400">
          <i class="fas fa-inbox text-4xl mb-4 block"></i>
          No active positions
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = "";

  Object.entries(positions).forEach(([positionId, position]) => {
    const coin = position.coin;
    const currentPrice = marketData[coin]?.price || position.entry_price;
    const entryPrice = position.entry_price;
    const direction = position.direction;
    const leverage = position.leverage;

    let pnlPercent;
    if (direction === "LONG") {
      pnlPercent = (currentPrice - entryPrice) / entryPrice;
    } else {
      pnlPercent = (entryPrice - currentPrice) / entryPrice;
    }

    const pnlAmount = pnlPercent * position.notional_value;
    const duration = calculateDuration(position.entry_time);

    const pnlColor = pnlAmount >= 0 ? "text-green-400" : "text-red-400";
    const directionColor =
      direction === "LONG" ? "text-green-400" : "text-red-400";
    const directionIcon =
      direction === "LONG" ? "fa-arrow-up" : "fa-arrow-down";

    // Distance to SL/TP
    const slDistance =
      direction === "LONG"
        ? ((currentPrice - position.stop_loss) / currentPrice) * 100
        : ((position.stop_loss - currentPrice) / currentPrice) * 100;

    const tpDistance =
      direction === "LONG"
        ? ((position.take_profit - currentPrice) / currentPrice) * 100
        : ((currentPrice - position.take_profit) / currentPrice) * 100;

    // Market regime icon
    const regimeIcons = {
      TRENDING_UP: "🚀",
      TRENDING_DOWN: "📉",
      RANGING: "↔️",
      VOLATILE: "⚡",
      TRANSITIONAL: "🔄",
      UNKNOWN: "❓",
    };
    const regimeIcon = regimeIcons[position.market_regime] || "❓";

    tbody.innerHTML += `
      <tr class="border-b border-gray-700 hover:bg-gray-800">
        <td class="py-3 px-4 font-semibold">
          <span class="${directionColor}">
            <i class="fas ${directionIcon} mr-1"></i>${direction}
          </span>
          ${coin}
          <br><span class="text-xs text-gray-400">${leverage}x • ${
      position.duration_target || "SWING"
    }</span>
        </td>
        <td class="py-3 px-4">
          $${position.position_size.toFixed(2)}
          <br><span class="text-xs text-gray-400">$${position.notional_value.toLocaleString()}</span>
        </td>
        <td class="py-3 px-4">$${entryPrice.toLocaleString()}</td>
        <td class="py-3 px-4">$${currentPrice.toLocaleString()}</td>
        <td class="py-3 px-4 ${pnlColor} font-semibold">
          $${pnlAmount >= 0 ? "+" : ""}${pnlAmount.toFixed(2)}
          <br><span class="text-sm">(${pnlPercent >= 0 ? "+" : ""}${(
      pnlPercent * 100
    ).toFixed(1)}%)</span>
        </td>
        <td class="py-3 px-4">${duration}</td>
        <td class="py-3 px-4">
          $${position.stop_loss.toLocaleString()}
          <br><span class="text-xs ${
            slDistance < 1 ? "text-red-400" : "text-gray-400"
          }">${slDistance.toFixed(1)}%</span>
        </td>
        <td class="py-3 px-4">
          $${position.take_profit.toLocaleString()}
          <br><span class="text-xs ${
            tpDistance < 1 ? "text-green-400" : "text-gray-400"
          }">${tpDistance.toFixed(1)}%</span>
        </td>
        <td class="py-3 px-4 text-center">
          <span class="text-xs font-semibold px-2 py-1 rounded ${
            position.confidence >= 8
              ? "bg-green-900 text-green-400"
              : position.confidence >= 6
              ? "bg-yellow-900 text-yellow-400"
              : "bg-red-900 text-red-400"
          }">
            ${position.confidence || 5}/10
          </span>
        </td>
        <td class="py-3 px-4 text-center" title="${
          position.market_regime || "UNKNOWN"
        }">
          <span class="text-2xl">${regimeIcon}</span>
        </td>
      </tr>
    `;
  });
}

function updateLeverageTradeHistory(tradeHistory) {
  const tbody = document.getElementById("trades-body");

  if (tradeHistory.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" class="text-center py-8 text-gray-400">
          <i class="fas fa-chart-line text-4xl mb-4 block"></i>
          No trades yet - AI is analyzing market conditions
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = "";

  // Show last 10 trades
  tradeHistory
    .slice(-10)
    .reverse()
    .forEach((trade) => {
      const isClose = trade.action?.includes("CLOSE");
      const actionColor = trade.action?.includes("LONG")
        ? "text-green-400"
        : "text-red-400";
      const actionIcon = trade.action?.includes("LONG")
        ? "fa-arrow-up"
        : "fa-arrow-down";

      let pnlCell = "-";
      if (trade.pnl !== undefined && trade.pnl !== null) {
        const pnlColor = trade.pnl >= 0 ? "text-green-400" : "text-red-400";
        pnlCell = `<span class="${pnlColor} font-semibold">$${
          trade.pnl >= 0 ? "+" : ""
        }${trade.pnl.toFixed(2)}</span>`;
        if (trade.pnl_percent) {
          pnlCell += `<br><span class="text-xs ${pnlColor}">(${
            trade.pnl_percent >= 0 ? "+" : ""
          }${trade.pnl_percent.toFixed(1)}%)</span>`;
        }
      }

      // Risk/Reward calculation if available
      let rrCell = "-";
      if (trade.stop_loss && trade.take_profit && trade.price && !isClose) {
        const slRisk = Math.abs(trade.price - trade.stop_loss) / trade.price;
        const tpReward =
          Math.abs(trade.take_profit - trade.price) / trade.price;
        const rr = tpReward / slRisk;
        rrCell = `<span class="text-xs text-gray-400">1:${rr.toFixed(
          1
        )}</span>`;
      }

      tbody.innerHTML += `
      <tr class="border-b border-gray-700">
        <td class="py-3 px-4 text-sm">${new Date(
          trade.time
        ).toLocaleTimeString()}</td>
        <td class="py-3 px-4 font-semibold">${trade.coin}</td>
        <td class="py-3 px-4 ${actionColor}">
          <i class="fas ${actionIcon} mr-1"></i>${trade.action || "-"}
        </td>
        <td class="py-3 px-4">$${trade.price?.toLocaleString() || "-"}</td>
        <td class="py-3 px-4">${
          trade.position_size ? `$${trade.position_size.toFixed(2)}` : "-"
        }</td>
        <td class="py-3 px-4">${
          trade.notional_value
            ? `$${trade.notional_value.toLocaleString()}`
            : "-"
        }</td>
        <td class="py-3 px-4">${pnlCell}</td>
        <td class="py-3 px-4">${rrCell}</td>
      </tr>
    `;
    });
}

function updateLearningMetrics(metrics) {
  let metricsContainer = document.getElementById("learning-metrics");
  if (!metricsContainer) {
    const mainContainer = document.querySelector(".container");
    const metricsHTML = `
      <div class="mt-8" id="learning-metrics-section">
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 shadow-xl">
          <h2 class="text-xl font-bold mb-6 text-blue-400">
            <i class="fas fa-brain mr-2"></i>AI Learning Metrics
          </h2>
          <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="learning-metrics">
            <!-- Metrics will be populated here -->
          </div>
          <div class="mt-4" id="regime-performance">
            <!-- Regime performance will be populated here -->
          </div>
        </div>
      </div>
    `;
    mainContainer.insertAdjacentHTML("beforeend", metricsHTML);
    metricsContainer = document.getElementById("learning-metrics");
  }

  if (metrics) {
    // Calculate Sharpe ratio if we have trade history
    let sharpeRatio = 0;
    if (
      window.data &&
      window.data.trade_history &&
      window.data.trade_history.length > 5
    ) {
      const returns = window.data.trade_history
        .filter((t) => t.pnl_percent !== undefined)
        .map((t) => t.pnl_percent);

      if (returns.length > 1) {
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance =
          returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) /
          returns.length;
        const stdDev = Math.sqrt(variance);
        sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;
      }
    }

    metricsContainer.innerHTML = `
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold ${
          metrics.win_rate >= 50 ? "text-green-400" : "text-red-400"
        }">${metrics.win_rate?.toFixed(1) || 0}%</p>
        <p class="text-gray-400 text-sm">Win Rate</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-yellow-400">${
          metrics.best_leverage || 10
        }x</p>
        <p class="text-gray-400 text-sm">Optimal Leverage</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-blue-400">${
          metrics.total_trades || 0
        }</p>
        <p class="text-gray-400 text-sm">Total Trades</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold text-purple-400">
          ${(
            (metrics.avg_profit || 0) - Math.abs(metrics.avg_loss || 0)
          ).toFixed(2)}
        </p>
        <p class="text-gray-400 text-sm">Avg P&L</p>
      </div>
      <div class="text-center bg-gray-700 rounded p-3">
        <p class="text-2xl font-bold ${
          sharpeRatio >= 1
            ? "text-green-400"
            : sharpeRatio >= 0
            ? "text-yellow-400"
            : "text-red-400"
        }">
          ${sharpeRatio.toFixed(2)}
        </p>
        <p class="text-gray-400 text-sm">Sharpe Ratio</p>
      </div>
    `;

    // Show regime performance if available
    if (
      metrics.regime_performance &&
      Object.keys(metrics.regime_performance).length > 0
    ) {
      const regimePerf = document.getElementById("regime-performance");
      if (regimePerf) {
        let perfHTML =
          '<h3 class="text-sm font-semibold text-gray-400 mb-2 mt-4">Performance by Market Regime</h3><div class="grid grid-cols-2 md:grid-cols-4 gap-2">';

        for (const [regime, data] of Object.entries(
          metrics.regime_performance
        )) {
          const avgPnl = data.avg_pnl || 0;
          const pnlColor = avgPnl >= 0 ? "text-green-400" : "text-red-400";
          perfHTML += `
            <div class="bg-gray-700 rounded p-2 text-center">
              <p class="text-xs text-gray-400">${regime.replace("_", " ")}</p>
              <p class="${pnlColor} font-bold">${avgPnl.toFixed(1)}%</p>
              <p class="text-xs text-gray-500">${data.count} trades</p>
            </div>
          `;
        }

        perfHTML += "</div>";
        regimePerf.innerHTML = perfHTML;
      }
    }
  }
}

function updatePortfolioChart(data) {
  if (!portfolioChart) return;

  const positions = Object.values(data.positions);
  const labels = ["Cash"];
  const values = [data.balance];
  const colors = ["#60a5fa"];

  positions.forEach((position, index) => {
    labels.push(`${position.coin} ${position.direction}`);
    values.push(position.position_size);
    colors.push(getColorForIndex(index));
  });

  portfolioChart.data.labels = labels;
  portfolioChart.data.datasets[0].data = values;
  portfolioChart.data.datasets[0].backgroundColor = colors;
  portfolioChart.update();
}

function updatePerformanceChart(data) {
  if (!performanceChart) return;

  // Update with latest portfolio value
  const currentTime = new Date().toLocaleTimeString();

  // Keep only last 20 data points
  if (performanceChart.data.labels.length >= 20) {
    performanceChart.data.labels.shift();
    performanceChart.data.datasets[0].data.shift();
  }

  performanceChart.data.labels.push(currentTime);
  performanceChart.data.datasets[0].data.push(data.total_value);
  performanceChart.update();
}

function calculateDuration(entryTime) {
  const entry = new Date(entryTime);
  const now = new Date();
  const diff = now - entry;
  const hours = Math.floor(diff / 3600000);
  const minutes = Math.floor((diff % 3600000) / 60000);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else {
    return `${minutes}m`;
  }
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
