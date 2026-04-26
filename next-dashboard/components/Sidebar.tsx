"use client";
import { motion } from "framer-motion";
import { 
  Activity, 
  BarChart3, 
  Settings, 
  HelpCircle, 
  ShieldCheck, 
  Download,
  Users,
  LayoutDashboard
} from "lucide-react";

type NavItem = {
  id: string;
  label: string;
  icon: any;
};

const NAV_ITEMS: NavItem[] = [
  { id: "overview", label: "Dashboard", icon: LayoutDashboard },
  { id: "subject", label: "Subject Intelligence", icon: Users },
  { id: "analytics", label: "Behavioral Analytics", icon: BarChart3 },
  { id: "security", label: "System Security", icon: ShieldCheck },
  { id: "settings", label: "App Settings", icon: Settings },
];

export default function Sidebar({ activeTab, onTabChange }: { activeTab: string, onTabChange: (id: any) => void }) {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Activity size={24} color="#00f2fe" strokeWidth={3} />
            <span className="logo-text-short" style={{ fontSize: '1.2rem', letterSpacing: '0.05em' }}>GESTURAI</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.25rem' }}>
            <span style={{ fontSize: '0.6rem', padding: '0.2rem 0.4rem', background: 'rgba(0, 242, 254, 0.1)', color: '#00f2fe', borderRadius: '4px', fontWeight: 600, border: '1px solid rgba(0, 242, 254, 0.2)' }}>WATCH ONLY</span>
            <span style={{ fontSize: '0.6rem', padding: '0.2rem 0.4rem', background: 'rgba(52, 211, 153, 0.1)', color: '#34d399', borderRadius: '4px', fontWeight: 600, border: '1px solid rgba(52, 211, 153, 0.2)' }}>70.0% VAL ACC</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeTab === item.id ? "active" : ""}`}
            onClick={() => onTabChange(item.id)}
          >
            <item.icon size={18} />
            <span className="nav-label">{item.label}</span>
            {activeTab === item.id && (
              <motion.div
                layoutId="active-nav"
                className="active-indicator"
              />
            )}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="nav-item">
          <HelpCircle size={18} />
          <span className="nav-label">Support</span>
        </button>
        <div className="user-profile-compact">
          <div className="avatar">JD</div>
          <div className="user-info">
            <span className="username">John Doe</span>
            <span className="role">Industrial Admin</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
