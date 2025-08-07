let portfolioChart = null;

// Initialize dashboard
document.addEventListener("DOMContentLoaded", function () {
  initPortfolioChart();
  fetchData();

  // Update every 30 seconds
  setInterval(fetchData, 30000);
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

async function fetchData() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    updateSummaryCards(data);
    updateMarketData(data.market_data);
    updateLeveragePositions(data.positions, data.market_data);
    updateLeverageTradeHistory(data.trade_history);
    updatePortfolioChart(data);

    // Show learning metrics
    updateLearningMetrics(data.learning_metrics);
  } catch (error) {
    console.error("Error fetching data:", error);
    // Show connection status
    document.querySelector(".text-green-400").innerHTML =
      '<i class="fas fa-circle text-xs mr-1 text-red-400"></i><span class="text-sm text-red-400">OFFLINE</span>';
  }
}

function updateSummaryCards(data) {
  document.getElementById(
    "total-value"
  ).textContent = `$${data.total_value.toFixed(2)}`;

  const pnlElement = document.getElementById("total-pnl");
  const pnl = data.total_value - 1000;
  pnlElement.textContent = `$${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}`;
  pnlElement.className = `text-2xl font-bold ${
    pnl >= 0 ? "text-green-400" : "text-red-400"
  }`;

  document.getElementById("active-positions").textContent = Object.keys(
    data.positions
  ).length;
  document.getElementById("total-trades").textContent =
    data.trade_history.length;
}

function updateMarketData(marketData) {
  const container = document.getElementById("market-data");
  container.innerHTML = "";

  Object.entries(marketData).forEach(([coin, data]) => {
    const changeColor =
      data.change_24h >= 0 ? "text-green-400" : "text-red-400";
    const changeIcon = data.change_24h >= 0 ? "fa-arrow-up" : "fa-arrow-down";

    container.innerHTML += `
            <div class="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div class="flex items-center">
                    <div class="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center mr-4">
                        <span class="font-bold text-sm">${coin}</span>
                    </div>
                    <div>
                        <p class="font-semibold">${coin}</p>
                        <p class="text-gray-400 text-sm">$${data.price.toLocaleString()}</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="${changeColor} font-semibold">
                        <i class="fas ${changeIcon} mr-1"></i>
                        ${
                          data.change_24h >= 0 ? "+" : ""
                        }${data.change_24h.toFixed(2)}%
                    </p>
                    <p class="text-gray-400 text-sm">24h</p>
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
                <td colspan="8" class="text-center py-8 text-gray-400">
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
    const currentPrice = marketData[coin].price;
    const entryPrice = position.entry_price;
    const direction = position.direction;
    const leverage = position.leverage;

    // Calculate P&L for leverage position
    let pnlPercent;
    if (direction === "LONG") {
      pnlPercent = (currentPrice - entryPrice) / entryPrice;
    } else {
      // SHORT
      pnlPercent = (entryPrice - currentPrice) / entryPrice;
    }

    const pnlAmount = pnlPercent * position.notional_value;
    const duration = calculateDuration(position.entry_time);

    const pnlColor = pnlAmount >= 0 ? "text-green-400" : "text-red-400";
    const directionColor =
      direction === "LONG" ? "text-green-400" : "text-red-400";
    const directionIcon =
      direction === "LONG" ? "fa-arrow-up" : "fa-arrow-down";

    tbody.innerHTML += `
            <tr class="border-b border-gray-700">
                <td class="py-3 px-4 font-semibold">
                    <span class="${directionColor}">
                        <i class="fas ${directionIcon} mr-1"></i>${direction}
                    </span>
                    ${coin}
                    <br><span class="text-xs text-gray-400">${leverage}x Leverage</span>
                </td>
                <td class="py-3 px-4">
                    $${position.position_size.toFixed(2)}
                    <br><span class="text-xs text-gray-400">Notional: $${position.notional_value.toLocaleString()}</span>
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
                <td class="py-3 px-4">$${position.stop_loss.toLocaleString()}</td>
                <td class="py-3 px-4">$${position.take_profit.toLocaleString()}</td>
            </tr>
        `;
  });
}

function updateLeverageTradeHistory(tradeHistory) {
  const tbody = document.getElementById("trades-body");

  if (tradeHistory.length === 0) {
    tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center py-8 text-gray-400">
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
      const isOpen = trade.action.includes("OPEN");
      const isClose =
        trade.action.includes("CLOSE") ||
        trade.action.includes("STOP_LOSS") ||
        trade.action.includes("TAKE_PROFIT");

      let actionColor = "text-gray-400";
      let actionIcon = "fa-exchange-alt";
      let actionText = trade.action;

      if (trade.action.includes("LONG")) {
        actionColor = "text-green-400";
        actionIcon = "fa-arrow-up";
      } else if (trade.action.includes("SHORT")) {
        actionColor = "text-red-400";
        actionIcon = "fa-arrow-down";
      }

      let pnlCell = "-";
      if (trade.pnl !== undefined && trade.pnl !== null) {
        const pnlColor = trade.pnl >= 0 ? "text-green-400" : "text-red-400";
        pnlCell = `<span class="${pnlColor}">$${
          trade.pnl >= 0 ? "+" : ""
        }${trade.pnl.toFixed(2)}</span>`;
        if (trade.pnl_percent) {
          pnlCell += `<br><span class="text-xs ${pnlColor}">(${
            trade.pnl_percent >= 0 ? "+" : ""
          }${trade.pnl_percent.toFixed(1)}%)</span>`;
        }
      }

      const leverageInfo = trade.leverage ? `${trade.leverage}x` : "";
      const positionSize = trade.position_size
        ? `$${trade.position_size.toFixed(2)}`
        : "-";

      tbody.innerHTML += `
            <tr class="border-b border-gray-700">
                <td class="py-3 px-4 text-sm">${new Date(
                  trade.time
                ).toLocaleTimeString()}</td>
                <td class="py-3 px-4 font-semibold">
                    ${trade.coin}
                    ${
                      leverageInfo
                        ? `<br><span class="text-xs text-gray-400">${leverageInfo}</span>`
                        : ""
                    }
                </td>
                <td class="py-3 px-4 ${actionColor}">
                    <i class="fas ${actionIcon} mr-1"></i>${actionText}
                </td>
                <td class="py-3 px-4">$${trade.price.toLocaleString()}</td>
                <td class="py-3 px-4">${positionSize}</td>
                <td class="py-3 px-4">
                    ${
                      trade.notional_value
                        ? `$${trade.notional_value.toLocaleString()}`
                        : positionSize
                    }
                </td>
                <td class="py-3 px-4">${pnlCell}</td>
            </tr>
        `;
    });
}

function updateLearningMetrics(metrics) {
  // Add learning metrics display if not exists
  let metricsContainer = document.getElementById("learning-metrics");
  if (!metricsContainer) {
    // Create learning metrics section
    const mainContainer = document.querySelector(".container");
    const metricsHTML = `
      <div class="mt-8" id="learning-metrics-section">
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 class="text-xl font-bold mb-6 text-blue-400">
            <i class="fas fa-brain mr-2"></i>AI Learning Metrics
          </h2>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="learning-metrics">
            <!-- Metrics will be populated here -->
          </div>
        </div>
      </div>
    `;
    mainContainer.insertAdjacentHTML("beforeend", metricsHTML);
    metricsContainer = document.getElementById("learning-metrics");
  }

  if (metrics) {
    metricsContainer.innerHTML = `
      <div class="text-center">
        <p class="text-2xl font-bold text-green-400">${
          metrics.win_rate?.toFixed(1) || 0
        }%</p>
        <p class="text-gray-400 text-sm">Win Rate</p>
      </div>
      <div class="text-center">
        <p class="text-2xl font-bold text-yellow-400">${
          metrics.best_leverage || 10
        }x</p>
        <p class="text-gray-400 text-sm">Best Leverage</p>
      </div>
      <div class="text-center">
        <p class="text-2xl font-bold text-blue-400">${
          metrics.total_trades || 0
        }</p>
        <p class="text-gray-400 text-sm">Total Trades</p>
      </div>
      <div class="text-center">
        <p class="text-2xl font-bold text-purple-400">$${
          metrics.avg_profit?.toFixed(2) || "0.00"
        }</p>
        <p class="text-gray-400 text-sm">Avg Profit</p>
      </div>
    `;
  }
}

function updatePortfolioChart(data) {
  const labels = ["Cash"];
  const values = [data.balance];
  const colors = ["#60a5fa"];

  Object.entries(data.positions).forEach(([positionId, position]) => {
    const coin = position.coin;
    const currentPrice = data.market_data[coin].price;
    const entryPrice = position.entry_price;
    const direction = position.direction;

    // Calculate current position value
    let pnlPercent;
    if (direction === "LONG") {
      pnlPercent = (currentPrice - entryPrice) / entryPrice;
    } else {
      pnlPercent = (entryPrice - currentPrice) / entryPrice;
    }

    const currentValue =
      position.position_size + pnlPercent * position.notional_value;
    labels.push(`${direction} ${coin}`);
    values.push(Math.max(0, currentValue)); // Don't show negative values in chart
    colors.push(getRandomColor());
  });

  portfolioChart.data.labels = labels;
  portfolioChart.data.datasets[0].data = values;
  portfolioChart.data.datasets[0].backgroundColor = colors;
  portfolioChart.update();
}

function calculateDuration(entryTime) {
  const now = new Date();
  const entry = new Date(entryTime);
  const diff = Math.floor((now - entry) / 1000);

  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

function getRandomColor() {
  const colors = [
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#84cc16",
  ];
  return colors[Math.floor(Math.random() * colors.length)];
}
