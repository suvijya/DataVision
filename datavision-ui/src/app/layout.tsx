import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import '@fortawesome/fontawesome-free/css/all.min.css';
import './globals.css';
import { SessionProvider } from '@/hooks/useSession';

const inter = Inter({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'DataVision Assistant — AI-Powered Data Analysis',
  description:
    'Transform your CSV data into insights using natural language queries. Upload, explore, and analyze with AI.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.variable}>
        <SessionProvider>{children}</SessionProvider>
      </body>
    </html>
  );
}
