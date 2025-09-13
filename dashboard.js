// Advanced Trading Bot Dashboard
let performanceChart = null;
let portfolioHistory = [];
let lastUpdateTime = null;
let updateInterval = null;

// Dashboard State
const dashboardState = {
  isConnected: true,
  sections: new Map(),
  lastData: null,
  errorCount: 0,
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", function () {
  console.log("🚀 Initializing Advanced Trading Dashboard...");

  initializeComponents();
  initializeEventListeners();
  startDataFetching();

  // Initialize sections state
  const toggleButtons = document.querySelectorAll(".toggle-button");
  toggleButtons.forEach((button) => {
    const sectionId = button.getAttribute("data-toggle");
    const section = document.getElementById(sectionId);
    const isActive = button.classList.contains("active");

    dashboardState.sections.set(sectionId, isActive);

    if (!isActive && section) {
      section.classList.add("collapsed");
    }
  });
});

function initializeComponents() {
  initPerformanceChart();
  updateLastUpdateTime();
}

function initializeEventListeners() {
  // Toggle button functionality
  document.querySelectorAll(".toggle-button").forEach((button) => {
    button.addEventListener("click", function () {
      const sectionId = this.getAttribute("data-toggle");
      const section = document.getElementById(sectionId);

      if (!section) return;

      this.classList.toggle("active");
      const isActive = this.classList.contains("active");

      dashboardState.sections.set(sectionId, isActive);

      if (isActive) {
        section.classList.remove("collapsed");
      } else {
        section.classList.add("collapsed");
      }
    });
  });

  // Handle connection events
  window.addEventListener("online", () => updateConnectionStatus(true));
  window.addEventListener("offline", () => updateConnectionStatus(false));
}

function initPerformanceChart() {
  const canvas = document.getElementById("performance-chart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  performanceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Portfolio Value",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "rgba(15, 23, 42, 0.9)",
          titleColor: "#f8fafc",
          bodyColor: "#cbd5e1",
          borderColor: "#3b82f6",
          borderWidth: 1,
        },
      },
      interaction: {
        mode: "nearest",
        axis: "x",
        intersect: false,
      },
      scales: {
        x: {
          display: false,
          grid: { display: false },
        },
        y: {
          display: false,
          grid: { display: false },
        },
      },
    },
  });
}

function startDataFetching() {
  // Initial fetch
  fetchData();

  // Set up interval
  if (updateInterval) clearInterval(updateInterval);
  updateInterval = setInterval(fetchData, 15000); // 15 seconds
}

async function fetchData() {
  try {
    const response = await fetch("/api/status");

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // Update dashboard with new data
    updateDashboard(data);
    updateConnectionStatus(true);
    dashboardState.errorCount = 0;
    dashboardState.lastData = data;
  } catch (error) {
    console.error("❌ Data fetch error:", error);
    dashboardState.errorCount++;
    updateConnectionStatus(false, error.message);

    // Try to use last good data if available
    if (dashboardState.lastData && dashboardState.errorCount < 5) {
      console.log("📊 Using cached data during connection issues");
    }
  }

  updateLastUpdateTime();
}

function updateDashboard(data) {
  try {
    updatePortfolioMetrics(data);
    updateActivePositions(data.positions, data.market_data);
    updateTradeHistory(data.trade_history);
    updateMarketData(data.market_data);
    updatePerformanceChart(data);
    updateLearningMetrics(data.learning_metrics);
    updateLearningProgress(data.learning_progress);
    updateCostTracking(data.cost_tracking);
  } catch (error) {
    console.error("❌ Dashboard update error:", error);
  }
}

function updatePortfolioMetrics(data) {
  // Calculate metrics
  const totalValue = data.total_value || 0;
  const balance = data.balance || 0;
  const positions = data.positions || {};
  const activePositionsCount = Object.keys(positions).length;

  // Calculate total unrealized P&L
  let totalUnrealizedPnL = 0;
  let totalMarginUsed = 0;

  Object.values(positions).forEach((position) => {
    totalUnrealizedPnL += position.pnl || 0;
    totalMarginUsed +=
      (position.notional || 0) / Math.max(1, position.leverage || 1);
  });

  const marginUsagePercent =
    balance > 0 ? (totalMarginUsed / balance) * 100 : 0;

  // Update UI elements
  updateElement("total-value", `$${totalValue.toLocaleString()}`);
  updateElement("balance", `$${balance.toLocaleString()}`);
  updateElement("active-positions", activePositionsCount.toString());

  // P&L with color coding
  const pnlElement = document.getElementById("total-pnl");
  const pnlPercentElement = document.getElementById("pnl-percentage");

  if (pnlElement) {
    pnlElement.textContent = `$${
      totalUnrealizedPnL >= 0 ? "+" : ""
    }${totalUnrealizedPnL.toFixed(2)}`;
    pnlElement.className = `text-2xl font-bold ${
      totalUnrealizedPnL >= 0 ? "text-green-400" : "text-red-400"
    }`;
  }

  if (pnlPercentElement && balance > 0) {
    const pnlPercent = (totalUnrealizedPnL / balance) * 100;
    pnlPercentElement.textContent = `${
      pnlPercent >= 0 ? "+" : ""
    }${pnlPercent.toFixed(2)}%`;
    pnlPercentElement.className = `text-xs ${
      pnlPercent >= 0 ? "text-green-400" : "text-red-400"
    }`;
  }

  // Margin usage
  updateElement("margin-usage", `${marginUsagePercent.toFixed(1)}%`);
  const marginProgress = document.getElementById("margin-progress");
  if (marginProgress) {
    marginProgress.style.width = `${Math.min(marginUsagePercent, 100)}%`;
    marginProgress.style.backgroundColor =
      marginUsagePercent > 80
        ? "#ef4444"
        : marginUsagePercent > 60
        ? "#f59e0b"
        : "#10b981";
  }

  // Update learning metrics if available
  if (data.learning_metrics) {
    const winRate = data.learning_metrics.win_rate || 0;
    const totalTrades = data.learning_metrics.total_trades || 0;

    updateElement("win-rate", `${winRate.toFixed(1)}%`);
    updateElement("total-trades", `${totalTrades} trades`);

    const winRateElement = document.getElementById("win-rate");
    if (winRateElement) {
      winRateElement.className = `text-2xl font-bold ${
        winRate >= 50 ? "text-green-400" : "text-red-400"
      }`;
    }
  }

  // Update positions count badge
  updateElement("positions-count", `${activePositionsCount} Open`);
}

function updateActivePositions(positions, marketData) {
  const tbody = document.getElementById("positions-tbody");
  if (!tbody) return;

  if (!positions || Object.keys(positions).length === 0) {
    tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center py-8 text-gray-500">
                    <i class="fas fa-chart-line text-2xl mb-2 block opacity-50"></i>
                    <span class="text-sm">No active positions</span>
                </td>
            </tr>`;
    return;
  }

  let rowsHTML = "";

  Object.entries(positions).forEach(([symbol, position]) => {
    const coin = position.coin;
    const currentPrice = position.mark_price;
    const entryPrice = position.entry_price;
    const direction = position.direction;
    const leverage = position.leverage || 1;
    const size = Math.abs(position.size || 0);
    const pnl = position.pnl || 0;

    // Calculate P&L percentage
    let pnlPercent = 0;
    if (direction === "LONG") {
      pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100;
    } else {
      pnlPercent = ((entryPrice - currentPrice) / entryPrice) * 100;
    }

    const pnlColor = pnl >= 0 ? "text-green-400" : "text-red-400";
    const directionClass =
      direction === "LONG" ? "long-indicator" : "short-indicator";

    rowsHTML += `
            <tr class="table-row">
                <td class="py-3 px-2">
                    <div class="flex items-center space-x-2">
                        <div class="w-6 h-6 bg-gradient-to-br from-blue-500 to-blue-700 rounded flex items-center justify-center">
                            <span class="text-xs font-bold text-white">${coin.slice(
                              0,
                              2
                            )}</span>
                        </div>
                        <span class="font-medium">${coin}</span>
                    </div>
                </td>
                <td class="py-3 px-2">
                    <span class="trading-indicator ${directionClass}">${direction}</span>
                </td>
                <td class="py-3 px-2 text-right font-mono text-sm">${size.toFixed(
                  4
                )}</td>
                <td class="py-3 px-2 text-right font-mono text-sm">$${entryPrice.toLocaleString()}</td>
                <td class="py-3 px-2 text-right font-mono text-sm">$${currentPrice.toLocaleString()}</td>
                <td class="py-3 px-2 text-right ${pnlColor}">
                    <div class="font-bold">$${pnl >= 0 ? "+" : ""}${pnl.toFixed(
      2
    )}</div>
                    <div class="text-xs">${
                      pnlPercent >= 0 ? "+" : ""
                    }${pnlPercent.toFixed(1)}%</div>
                </td>
                <td class="py-3 px-2 text-right font-bold">${leverage}x</td>
            </tr>`;
  });

  tbody.innerHTML = rowsHTML;
}

function updateTradeHistory(tradeHistory) {
  const tbody = document.getElementById("trades-tbody");
  if (!tbody) return;

  if (!tradeHistory || tradeHistory.length === 0) {
    tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center py-8 text-gray-500">
                    <i class="fas fa-clock text-2xl mb-2 block opacity-50"></i>
                    <span class="text-sm">No trades yet</span>
                </td>
            </tr>`;
    return;
  }

  let rowsHTML = "";

  // Get last 20 trades, most recent first
  const recentTrades = tradeHistory.slice(-20).reverse();

  recentTrades.forEach((trade) => {
    // Handle different timestamp formats from your Python code
    let tradeTime;
    try {
      if (trade.timestamp) {
        tradeTime = new Date(trade.timestamp);
      } else if (trade.time) {
        tradeTime = new Date(trade.time);
      } else if (trade.created_at) {
        tradeTime = new Date(trade.created_at);
      } else {
        tradeTime = new Date();
      }
    } catch (e) {
      tradeTime = new Date();
    }

    const timeString = tradeTime.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    const dateString = tradeTime.toLocaleDateString();

    // Determine action and color
    const action =
      trade.action ||
      `${trade.direction || "UNKNOWN"} ${
        trade.pnl !== undefined && trade.pnl !== null ? "CLOSE" : "OPEN"
      }`;
    const isLong = action.includes("LONG");
    const isShort = action.includes("SHORT");
    const actionColor = isLong
      ? "text-green-400"
      : isShort
      ? "text-red-400"
      : "text-gray-400";
    const actionIcon = isLong
      ? "fa-arrow-up"
      : isShort
      ? "fa-arrow-down"
      : "fa-exchange-alt";

    // Handle P&L display
    let pnlDisplay = "--";
    let pnlColor = "text-gray-400";

    if (trade.pnl !== undefined && trade.pnl !== null) {
      const pnl = parseFloat(trade.pnl);
      if (!isNaN(pnl)) {
        pnlDisplay = `$${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}`;
        pnlColor = pnl >= 0 ? "text-green-400" : "text-red-400";

        // Add percentage if available
        if (trade.pnl_percent !== undefined && trade.pnl_percent !== null) {
          const pnlPercent = parseFloat(trade.pnl_percent);
          if (!isNaN(pnlPercent)) {
            pnlDisplay += `<br><span class="text-xs opacity-70">${
              pnlPercent >= 0 ? "+" : ""
            }${pnlPercent.toFixed(1)}%</span>`;
          }
        }
      }
    }

    // Confidence badge
    const confidence = trade.confidence || 5;
    const confColor =
      confidence >= 8
        ? "badge-success"
        : confidence >= 6
        ? "badge-warning"
        : "badge-danger";

    rowsHTML += `
            <tr class="table-row" title="${dateString}">
                <td class="py-3 px-2 text-xs font-mono text-gray-400">${timeString}</td>
                <td class="py-3 px-2">
                    <div class="flex items-center space-x-2">
                        <div class="w-5 h-5 bg-gradient-to-br from-blue-500 to-blue-700 rounded flex items-center justify-center">
                            <span class="text-xs font-bold text-white">${(
                              trade.coin || "XX"
                            ).slice(0, 2)}</span>
                        </div>
                        <span class="font-medium text-sm">${
                          trade.coin || "UNKNOWN"
                        }</span>
                    </div>
                </td>
                <td class="py-3 px-2">
                    <span class="${actionColor} flex items-center text-sm">
                        <i class="fas ${actionIcon} mr-1"></i>
                        <span class="text-xs">${action}</span>
                    </span>
                </td>
                <td class="py-3 px-2 text-right font-mono text-sm">
                    ${
                      trade.price
                        ? `$${parseFloat(trade.price).toLocaleString()}`
                        : "--"
                    }
                </td>
                <td class="py-3 px-2 text-right ${pnlColor} text-sm">${pnlDisplay}</td>
                <td class="py-3 px-2 text-center">
                    <span class="badge ${confColor} text-xs">${confidence}/10</span>
                </td>
            </tr>`;
  });

  tbody.innerHTML = rowsHTML;
}

function updateMarketData(marketData) {
  const container = document.getElementById("market-data");
  if (!container) return;

  if (!marketData || Object.keys(marketData).length === 0) {
    container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <i class="fas fa-sync fa-spin text-xl mb-2 block"></i>
                <span class="text-sm">Loading market data...</span>
            </div>`;
    return;
  }

  let marketHTML = "";

  // Sort by 24h change percentage (highest first)
  const sortedCoins = Object.entries(marketData).sort(
    ([, a], [, b]) => (b.change_24h || 0) - (a.change_24h || 0)
  );

  sortedCoins.forEach(([coin, data]) => {
    const change = data.change_24h || 0;
    const changeColor = change >= 0 ? "text-green-400" : "text-red-400";
    const changeIcon = change >= 0 ? "fa-arrow-up" : "fa-arrow-down";
    const price = data.price || 0;
    const volume = data.volume || 0;

    // Volume indicator
    const volumeLabel =
      volume > 100000000 ? "🔥 Hot" : volume > 50000000 ? "📈 Good" : "📊 Low";

    marketHTML += `
            <div class="flex items-center justify-between p-3 bg-gray-800/20 rounded-lg hover:bg-gray-800/40 transition-all duration-200 border border-gray-700/20 hover:border-blue-500/30">
                <div class="flex items-center space-x-3">
                    <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg flex items-center justify-center shadow-lg">
                        <span class="font-bold text-xs text-white">${coin.slice(
                          0,
                          2
                        )}</span>
                    </div>
                    <div>
                        <p class="font-semibold text-sm">${coin}/USDT</p>
                        <p class="text-gray-400 text-xs">$${price.toLocaleString()}</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="${changeColor} font-semibold text-sm flex items-center justify-end">
                        <i class="fas ${changeIcon} text-xs mr-1"></i>
                        ${change >= 0 ? "+" : ""}${change.toFixed(2)}%
                    </p>
                    <p class="text-gray-400 text-xs">
                        ${volumeLabel} ${(volume / 1000000).toFixed(1)}M
                    </p>
                </div>
            </div>`;
  });

  container.innerHTML = marketHTML;
}

function updatePerformanceChart(data) {
  if (!performanceChart) return;

  const currentTime = new Date();
  const currentValue = data.total_value || 0;

  // Add to portfolio history
  portfolioHistory.push({
    time: currentTime.toLocaleTimeString(),
    value: currentValue,
    timestamp: currentTime.getTime(),
  });

  // Keep only last 50 points
  if (portfolioHistory.length > 50) {
    portfolioHistory = portfolioHistory.slice(-50);
  }

  // Update chart
  performanceChart.data.labels = portfolioHistory.map((h) => h.time);
  performanceChart.data.datasets[0].data = portfolioHistory.map((h) => h.value);
  performanceChart.update("none"); // No animation for smoother updates
}

function updateLearningMetrics(metrics) {
  if (!metrics) return;

  const avgHoldTime = calculateAverageHoldTime();
  const bestLeverage = metrics.best_leverage || 15;

  updateElement("avg-hold-time", avgHoldTime);
  updateElement("best-leverage", `${bestLeverage}x`);
}

function updateCostTracking(costData) {
  if (!costData) return;

  const current = costData.current || {};
  const projections = costData.projections || {};

  updateElement("openai-cost", `$${(current.openai || 0).toFixed(4)}`);
  updateElement("railway-cost", `$${(current.railway || 0).toFixed(4)}`);
  updateElement(
    "monthly-projection",
    `$${(projections.monthly?.total || 0).toFixed(2)}`
  );
}

// Utility Functions
function updateElement(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function updateLearningProgress(learningData) {
  if (!learningData) return;

  const {
    is_learning_phase,
    total_learning_trades,
    min_required_trades,
    remaining_trades,
    progress_percent,
    learning_win_rate,
    ml_initialized,
    ml_ready,
    bootstrap_complete,
    retry_count,
    max_retries,
  } = learningData;

  // Update progress bar
  updateElement(
    "learning-progress-text",
    `${total_learning_trades}/${min_required_trades}`
  );
  updateElement("remaining-trades", `${remaining_trades} remaining`);
  updateElement(
    "learning-win-rate",
    `Win Rate: ${learning_win_rate.toFixed(1)}%`
  );

  const progressBar = document.getElementById("learning-progress-bar");
  if (progressBar) {
    progressBar.style.width = `${progress_percent}%`;
  }

  // Update badges
  const learningBadge = document.getElementById("learning-phase-badge");
  const mlStatusBadge = document.getElementById("ml-status-badge");

  if (is_learning_phase) {
    learningBadge.textContent = "LEARNING";
    learningBadge.className = "badge badge-warning";
    mlStatusBadge.textContent = "BOOTSTRAP";
    mlStatusBadge.className = "badge badge-info";
  } else {
    learningBadge.textContent = "COMPLETE";
    learningBadge.className = "badge badge-success";
    if (ml_ready) {
      mlStatusBadge.textContent = "ACTIVE";
      mlStatusBadge.className = "badge badge-success";
    } else {
      mlStatusBadge.textContent = "INITIALIZING";
      mlStatusBadge.className = "badge badge-warning";
    }
  }

  // Update status indicators
  updateElement(
    "current-mode",
    is_learning_phase ? "BOOTSTRAP" : "ML-ENHANCED"
  );

  if (ml_initialized) {
    updateElement("ml-system-status", "READY");
  } else if (bootstrap_complete) {
    updateElement("ml-system-status", `INIT ${retry_count}/${max_retries}`);
  } else {
    updateElement("ml-system-status", "WAITING");
  }

  // Update ML activation status
  const activationIndicator = document.getElementById(
    "ml-activation-indicator"
  );
  const activationText = document.getElementById("ml-activation-text");
  const activationMessage = document.getElementById("ml-activation-message");

  if (ml_ready) {
    activationIndicator.className =
      "w-2 h-2 rounded-full bg-green-400 animate-pulse";
    activationText.textContent = "ACTIVE";
    activationText.className = "text-xs font-bold text-green-400";
    activationMessage.textContent = "ML system active and trading";
  } else if (bootstrap_complete && !ml_initialized) {
    activationIndicator.className =
      "w-2 h-2 rounded-full bg-yellow-400 animate-pulse";
    activationText.textContent = "INITIALIZING";
    activationText.className = "text-xs font-bold text-yellow-400";
    activationMessage.textContent = `Initializing ML models (${retry_count}/${max_retries})...`;
  } else {
    activationIndicator.className = "w-2 h-2 rounded-full bg-blue-400";
    activationText.textContent = "LEARNING";
    activationText.className = "text-xs font-bold text-blue-400";
    activationMessage.textContent = `${remaining_trades} more trades needed for ML activation`;
  }
}

function updateConnectionStatus(isConnected, errorMessage = "") {
  const statusElement = document.getElementById("connection-status");
  if (!statusElement) return;

  dashboardState.isConnected = isConnected;

  if (isConnected) {
    statusElement.innerHTML = `
            <div class="status-indicator status-live"></div>
            <span class="text-green-400 font-medium">CONNECTED</span>`;
  } else {
    statusElement.innerHTML = `
            <div class="status-indicator status-offline"></div>
            <span class="text-red-400 font-medium">OFFLINE</span>`;

    if (errorMessage) {
      console.error("Connection error:", errorMessage);
    }
  }
}

function updateLastUpdateTime() {
  const element = document.getElementById("last-update");
  if (element) {
    const now = new Date();
    element.textContent = `Last update: ${now.toLocaleTimeString()}`;
  }
  lastUpdateTime = Date.now();
}

function calculateAverageHoldTime() {
  // This would need to be calculated from trade history
  // For now, return a placeholder
  return "--";
}

// Error handling and cleanup
window.addEventListener("beforeunload", () => {
  if (updateInterval) {
    clearInterval(updateInterval);
  }
});

// Export for debugging
window.dashboardDebug = {
  state: dashboardState,
  portfolioHistory,
  fetchData,
  updateDashboard,
};
