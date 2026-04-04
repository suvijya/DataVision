'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import FileUpload from '@/components/FileUpload';
import DataPreview from '@/components/DataPreview';
import ChatSection from '@/components/Chat/ChatSection';
import LoadingOverlay from '@/components/ui/LoadingOverlay';
import { useSession } from '@/hooks/useSession';

export default function Home() {
  const { sessionId, dataPreview, isLoading, loadingText } = useSession();

  return (
    <div className="container">
      <Header />
      <main className="main">
        {!sessionId && <FileUpload />}
        {sessionId && dataPreview && (
          <>
            <DataPreview data={dataPreview} />
            <ChatSection />
          </>
        )}
      </main>
      <Footer />
      <LoadingOverlay text={loadingText} show={isLoading} />
    </div>
  );
}
