"use client";
import { Search, Bell, Cpu, HardDrive, Download } from "lucide-react";

export default function TopBar({ typedTitle }: { typedTitle: string }) {
  return (
    <div className="app-topbar">
      <div className="topbar-left">
        <h2 className="page-title">{typedTitle}<span className="cursor">|</span></h2>
      </div>

      <div className="topbar-center">
        <div className="search-box">
          <Search size={16} color="#475569" />
          <input type="text" placeholder="Search subjects, activities, or logs..." />
          <kbd className="search-kbd">/</kbd>
        </div>
      </div>

      <div className="topbar-right">
        {/* System Health Mocks */}
        <div className="system-health">
          <div className="health-item">
            <Cpu size={14} color="#94a3b8" />
            <span className="health-label">CPU</span>
            <div className="health-bar-mini">
              <div className="health-fill" style={{ width: '24%', background: '#34d399' }} />
            </div>
          </div>
          <div className="health-item">
            <HardDrive size={14} color="#94a3b8" />
            <span className="health-label">MEM</span>
            <div className="health-bar-mini">
              <div className="health-fill" style={{ width: '42%', background: '#fbbf24' }} />
            </div>
          </div>
        </div>

        <button className="icon-btn">
          <Bell size={20} />
          <span className="notification-dot" />
        </button>

        <button className="export-btn">
          <Download size={16} />
          <span>Export CSV</span>
        </button>
      </div>
    </div>
  );
}
