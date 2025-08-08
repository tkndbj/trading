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

    updateSummaryCards(data);
    updateMarketData(data.market_data);
    updateLeveragePositions(
      data.positions,
      data.market_data,
      data.position_pnls
    );
    updateLeverageTradeHistory(data.trade_history);
    updatePortfolioChart(data);
    updatePerformanceChart(data);
    updateLearningMetrics(data.learning_metrics);
    updateCostTracking(data.cost_tracking);
    updateMarketRegimes(data.positions, data.market_data);
    updateTradingModeIndicator(data);

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

function updateTradingModeIndicator(data) {
  const header = document.querySelector("h1");
  const tradingMode = data.trading_mode || "PAPER";
  const modeColor = tradingMode === "REAL" ? "text-red-400" : "text-blue-400";
  const modeIcon =
    tradingMode === "REAL" ? "fas fa-exclamation-triangle" : "fas fa-file-alt";

  let modeIndicator = document.getElementById("trading-mode-indicator");
  if (!modeIndicator) {
    modeIndicator = document.createElement("span");
    modeIndicator.id = "trading-mode-indicator";
    modeIndicator.className = `ml-3 text-sm font-bold ${modeColor}`;
    header.appendChild(modeIndicator);
  }

  modeIndicator.innerHTML = `<i class="${modeIcon} mr-1"></i>${tradingMode} MODE`;
  modeIndicator.className = `ml-3 text-sm font-bold ${modeColor}`;
}

function updateSummaryCards(data) {
  // Use dynamic data from API
  const initialBalance = data.initial_balance || 1000;
  const currentBalance = data.current_balance || initialBalance;
  const totalValue = data.total_value || currentBalance;
  const totalPnl = data.total_pnl || 0;
  const totalPnlPercent = data.total_pnl_percent || 0;
  const realizedPnl = data.realized_pnl || 0;
  const unrealizedPnl = data.unrealized_pnl || 0;

  // Update total value
  document.getElementById("total-value").textContent = `$${totalValue.toFixed(
    2
  )}`;

  // Update P&L with breakdown
  const pnlElement = document.getElementById("total-pnl");
  pnlElement.innerHTML = `
    $${totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}
    <span class="text-xs font-normal opacity-75">(${
      totalPnlPercent >= 0 ? "+" : ""
    }${totalPnlPercent.toFixed(1)}%)</span>
  `;
  pnlElement.className = `text-xl font-bold ${
    totalPnl >= 0 ? "text-green-400" : "text-red-400"
  }`;

  // Add P&L breakdown tooltip/details
  pnlElement.title = `Realized: $${realizedPnl.toFixed(
    2
  )} | Unrealized: $${unrealizedPnl.toFixed(2)}`;

  // Update positions and trades count
  document.getElementById("active-positions").textContent = Object.keys(
    data.positions || {}
  ).length;

  document.getElementById("total-trades").textContent =
    data.performance_metrics?.total_trades || 0;

  // Show real balance information if available
  if (data.real_balance && data.trading_mode === "REAL") {
    const balanceCard = document
      .querySelector("#total-value")
      .closest(".compact-card");

    let realBalanceIndicator = balanceCard.querySelector(
      ".real-balance-indicator"
    );
    if (!realBalanceIndicator) {
      realBalanceIndicator = document.createElement("div");
      realBalanceIndicator.className =
        "real-balance-indicator text-xs text-green-400 mt-1";
      balanceCard.querySelector("div > div").appendChild(realBalanceIndicator);
    }
    realBalanceIndicator.innerHTML = `🔴 Live: $${data.real_balance.toFixed(
      2
    )}`;
  }

  // Update balance breakdown in portfolio card
  const portfolioCard = document
    .querySelector("#total-value")
    .closest(".compact-card");
  let balanceBreakdown = portfolioCard.querySelector(".balance-breakdown");
  if (!balanceBreakdown) {
    balanceBreakdown = document.createElement("div");
    balanceBreakdown.className = "balance-breakdown text-xs text-gray-400 mt-1";
    portfolioCard.querySelector("div > div").appendChild(balanceBreakdown);
  }

  const cashPercent =
    totalValue > 0 ? (currentBalance / totalValue) * 100 : 100;
  const positionsValue = totalValue - currentBalance;
  const positionsPercent = 100 - cashPercent;

  balanceBreakdown.innerHTML = `
    Cash: $${currentBalance.toFixed(2)} (${cashPercent.toFixed(1)}%)
    ${
      positionsValue > 0
        ? `<br>Positions: $${positionsValue.toFixed(
            2
          )} (${positionsPercent.toFixed(1)}%)`
        : ""
    }
  `;
}

function updateMarketRegimes(positions, marketData) {
  let regimeContainer = document.getElementById("market-regimes");
  if (!regimeContainer) {
    const marketSection = document.getElementById("market-data-section");
    if (!marketSection) return;

    const regimeHTML = `
      <div class="mt-4 pt-4 border-t border-gray-700/50">
        <h3 class="text-sm font-semibold text-gray-400 mb-3 flex items-center">
          <i class="fas fa-brain mr-2"></i>Market Regimes
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2" id="market-regimes">
        </div>
      </div>
    `;
    marketSection.insertAdjacentHTML("beforeend", regimeHTML);
    regimeContainer = document.getElementById("market-regimes");
  }

  const regimeCounts = {};
  Object.values(positions || {}).forEach((pos) => {
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
      <div class="bg-gray-700/30 rounded-lg p-2 text-center hover:bg-gray-700/50 transition-colors">
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
  } else {
    regimeContainer.innerHTML =
      '<p class="text-gray-400 text-sm col-span-full text-center">No active positions</p>';
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
            <div class="grid grid-cols-2 md:grid-cols-5 gap-3" id="current-costs">
            </div>
          </div>
          
          <div>
            <h3 class="text-base font-semibold text-gray-300 mb-3">Projected Costs</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3" id="cost-projections">
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

  const currentCosts = document.getElementById("current-costs");
  if (currentCosts) {
    const totalValue = window.data?.total_value || 0;
    const initialBalance = window.data?.initial_balance || 1000;
    const totalPnl = window.data?.total_pnl || 0;
    const netProfit = totalPnl - (costData.current?.total || 0);

    currentCosts.innerHTML = `
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-blue-400">$${
          costData.current?.openai?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-xs">OpenAI</p>
        <p class="text-xs text-gray-500">${
          costData.current?.api_calls || 0
        } calls</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-purple-400">$${
          costData.current?.railway?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-xs">Railway</p>
        <p class="text-xs text-gray-500">Hosting</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-yellow-400">$${
          costData.current?.total?.toFixed(4) || "0.0000"
        }</p>
        <p class="text-gray-400 text-xs">Total Cost</p>
        <p class="text-xs text-gray-500">All services</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold ${
          netProfit >= 0 ? "text-green-400" : "text-red-400"
        }">
          $${netProfit.toFixed(2)}
        </p>
        <p class="text-gray-400 text-xs">Net Profit</p>
        <p class="text-xs text-gray-500">After costs</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-cyan-400">
          ${
            costData.current?.total > 0 && totalPnl > 0
              ? (totalPnl / costData.current.total).toFixed(1)
              : totalPnl > 0
              ? "∞"
              : "0"
          }x
        </p>
        <p class="text-gray-400 text-xs">ROI</p>
        <p class="text-xs text-gray-500">Return on cost</p>
      </div>
    `;
  }

  const projections = document.getElementById("cost-projections");
  if (projections && costData.projections) {
    projections.innerHTML = `
      <div class="bg-gradient-to-br from-blue-900/50 to-blue-800/50 rounded-lg p-4 border border-blue-500/20">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Weekly Estimate</h4>
        <div class="space-y-1">
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
      </div>
      
      <div class="bg-gradient-to-br from-purple-900/50 to-purple-800/50 rounded-lg p-4 border border-purple-500/20">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Monthly Estimate</h4>
        <div class="space-y-1">
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
      </div>
      
      <div class="bg-gradient-to-br from-green-900/50 to-green-800/50 rounded-lg p-4 border border-green-500/20">
        <h4 class="text-sm font-semibold text-gray-300 mb-2">Cost Efficiency</h4>
        <div class="space-y-1">
          <p class="text-xl font-bold text-white">
            $${(
              (costData.current?.total || 0) /
              (costData.current?.api_calls || 1)
            ).toFixed(4)}
          </p>
          <p class="text-xs text-gray-300">Per API Call</p>
          <p class="text-xs text-gray-300 mt-2">
            ${
              costData.current?.api_calls
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

  const sortedCoins = Object.entries(marketData || {}).sort(
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

function updateLeveragePositions(positions, marketData, positionPnls) {
  const tbody = document.getElementById("positions-body");

  if (!positions || Object.keys(positions).length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="text-center py-12 text-gray-400">
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

  Object.entries(positions).forEach(([positionId, position]) => {
    const coin = position.coin;
    const currentPrice = marketData[coin]?.price || position.entry_price;
    const entryPrice = position.entry_price;
    const direction = position.direction;
    const leverage = position.leverage;

    // Use API-provided P&L if available, otherwise calculate
    let pnlData = positionPnls?.[positionId];
    if (!pnlData) {
      let pnlPercent;
      if (direction === "LONG") {
        pnlPercent = (currentPrice - entryPrice) / entryPrice;
      } else {
        pnlPercent = (entryPrice - currentPrice) / entryPrice;
      }
      pnlData = {
        pnl_amount: pnlPercent * position.notional_value,
        pnl_percent: pnlPercent * 100,
      };
    }

    const duration = calculateDuration(position.entry_time);

    const pnlColor =
      pnlData.pnl_amount >= 0 ? "text-green-400" : "text-red-400";
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
      <tr class="border-b border-gray-700/30 hover:bg-gray-800/30 transition-colors">
        <td class="py-3 px-3 font-semibold">
          <span class="${directionColor} flex items-center">
            <i class="fas ${directionIcon} mr-1"></i>${direction}
          </span>
          <span class="text-white">${coin}</span>
          <br><span class="text-xs text-gray-400">${leverage}x • ${
      position.duration_target || "SWING"
    }</span>
        </td>
        <td class="py-3 px-3">
          <span class="font-medium">$${position.position_size.toFixed(2)}</span>
          <br><span class="text-xs text-gray-400">$${position.notional_value.toLocaleString()}</span>
        </td>
        <td class="py-3 px-3 font-mono text-sm">$${entryPrice.toLocaleString()}</td>
        <td class="py-3 px-3 font-mono text-sm">$${currentPrice.toLocaleString()}</td>
        <td class="py-3 px-3 ${pnlColor} font-semibold">
          $${pnlData.pnl_amount >= 0 ? "+" : ""}${pnlData.pnl_amount.toFixed(2)}
          <br><span class="text-sm">(${
            pnlData.pnl_percent >= 0 ? "+" : ""
          }${pnlData.pnl_percent.toFixed(1)}%)</span>
        </td>
        <td class="py-3 px-3 text-sm">${duration}</td>
        <td class="py-3 px-3">
          <span class="font-mono text-sm">$${position.stop_loss.toLocaleString()}</span>
          <br><span class="text-xs ${
            slDistance < 1 ? "text-red-400" : "text-gray-400"
          }">${slDistance.toFixed(1)}%</span>
        </td>
        <td class="py-3 px-3">
          <span class="font-mono text-sm">$${position.take_profit.toLocaleString()}</span>
          <br><span class="text-xs ${
            tpDistance < 1 ? "text-green-400" : "text-gray-400"
          }">${tpDistance.toFixed(1)}%</span>
        </td>
        <td class="py-3 px-3 text-center">
          <span class="text-xs font-semibold px-2 py-1 rounded-full ${
            position.confidence >= 8
              ? "bg-green-900/50 text-green-400 border border-green-500/30"
              : position.confidence >= 6
              ? "bg-yellow-900/50 text-yellow-400 border border-yellow-500/30"
              : "bg-red-900/50 text-red-400 border border-red-500/30"
          }">
            ${position.confidence || 5}/10
          </span>
        </td>
        <td class="py-3 px-3 text-center" title="${
          position.market_regime || "UNKNOWN"
        }">
          <span class="text-xl">${regimeIcon}</span>
        </td>
      </tr>
    `;
  });
}

function updateLeverageTradeHistory(tradeHistory) {
  const tbody = document.getElementById("trades-body");

  if (!tradeHistory || tradeHistory.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" class="text-center py-12 text-gray-400">
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

      // Risk/Reward calculation
      let rrCell = "-";
      if (trade.stop_loss && trade.take_profit && trade.price && !isClose) {
        const slRisk = Math.abs(trade.price - trade.stop_loss) / trade.price;
        const tpReward =
          Math.abs(trade.take_profit - trade.price) / trade.price;
        const rr = tpReward / slRisk;
        rrCell = `<span class="text-xs text-gray-400 bg-gray-700/30 px-2 py-1 rounded-full">1:${rr.toFixed(
          1
        )}</span>`;
      } else if (isClose && trade.pnl && trade.position_size) {
        // For closed trades, show actual R:R achieved
        const returnPercent = Math.abs(trade.pnl_percent || 0);
        rrCell = `<span class="text-xs ${
          trade.pnl >= 0 ? "text-green-400" : "text-red-400"
        } bg-gray-700/30 px-2 py-1 rounded-full">${returnPercent.toFixed(
          1
        )}%</span>`;
      }

      const tradeTime = new Date(trade.time);
      const timeString = tradeTime.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      const dateString = tradeTime.toLocaleDateString();

      tbody.innerHTML += `
      <tr class="border-b border-gray-700/30 hover:bg-gray-800/30 transition-colors">
        <td class="py-3 px-3 text-sm font-mono">
          ${timeString}
          <br><span class="text-xs text-gray-500">${dateString}</span>
        </td>
        <td class="py-3 px-3 font-semibold">${trade.coin}</td>
        <td class="py-3 px-3 ${actionColor}">
          <span class="flex items-center">
            <i class="fas ${actionIcon} mr-1"></i>${trade.action || "-"}
          </span>
          ${
            trade.leverage
              ? `<br><span class="text-xs text-gray-400">${trade.leverage}x</span>`
              : ""
          }
        </td>
        <td class="py-3 px-3 font-mono text-sm">$${
          trade.price?.toLocaleString() || "-"
        }</td>
        <td class="py-3 px-3 text-sm">${
          trade.position_size ? `$${trade.position_size.toFixed(2)}` : "-"
        }</td>
        <td class="py-3 px-3 text-sm">${
          trade.notional_value
            ? `$${trade.notional_value.toLocaleString()}`
            : "-"
        }</td>
        <td class="py-3 px-3">${pnlCell}</td>
        <td class="py-3 px-3 text-center">${rrCell}</td>
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
          <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4" id="learning-metrics">
          </div>
          <div id="regime-performance">
          </div>
        </div>
      </div>
    `;
    mainContainer.insertAdjacentHTML("beforeend", metricsHTML);
  }

  const metricsContainer = document.getElementById("learning-metrics");
  if (metrics && metricsContainer) {
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
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold ${
          metrics.win_rate >= 50 ? "text-green-400" : "text-red-400"
        }">${metrics.win_rate?.toFixed(1) || 0}%</p>
        <p class="text-gray-400 text-xs">Win Rate</p>
      </div>
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold text-yellow-400">${
          metrics.best_leverage || 10
        }x</p>
        <p class="text-gray-400 text-xs">Optimal Leverage</p>
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
      <div class="text-center bg-gray-700/30 rounded-lg p-3">
        <p class="text-lg font-bold ${
          sharpeRatio >= 1
            ? "text-green-400"
            : sharpeRatio >= 0
            ? "text-yellow-400"
            : "text-red-400"
        }">
          ${sharpeRatio.toFixed(2)}
        </p>
        <p class="text-gray-400 text-xs">Sharpe Ratio</p>
      </div>
    `;

    if (
      metrics.regime_performance &&
      Object.keys(metrics.regime_performance).length > 0
    ) {
      const regimePerf = document.getElementById("regime-performance");
      if (regimePerf) {
        let perfHTML =
          '<h3 class="text-sm font-semibold text-gray-400 mb-3 mt-4 pt-4 border-t border-gray-700/50">Performance by Market Regime</h3><div class="grid grid-cols-2 md:grid-cols-4 gap-2">';

        for (const [regime, data] of Object.entries(
          metrics.regime_performance
        )) {
          const avgPnl = data.avg_pnl || 0;
          const pnlColor = avgPnl >= 0 ? "text-green-400" : "text-red-400";
          perfHTML += `
            <div class="bg-gray-700/30 rounded-lg p-2 text-center">
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

  const positions = Object.values(data.positions || {});
  const labels = ["Cash"];
  const values = [data.current_balance || 0];
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
