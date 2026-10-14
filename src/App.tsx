import { useState, useEffect } from 'react';
import { Hero } from './components/Hero';
import { Architecture } from './components/Architecture';
import { Benchmarks } from './components/Benchmarks';
import { Innovations } from './components/Innovations';
import { CodeExamples } from './components/CodeExamples';
import { GithubReadme } from './components/GithubReadme';
import { Navigation } from './components/Navigation';
import { Footer } from './components/Footer';
import { DownloadSection } from './components/DownloadSection';

function App() {
  const [activeSection, setActiveSection] = useState('hero');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.3 }
    );

    const sections = document.querySelectorAll('section[id]');
    sections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen grid-bg">
      <Navigation activeSection={activeSection} />
      <main>
        <Hero />
        <section id="download" className="py-16 px-4 sm:px-6 lg:px-8">
          <DownloadSection />
        </section>
        <Architecture />
        <Benchmarks />
        <Innovations />
        <CodeExamples />
        <GithubReadme />
      </main>
      <Footer />
    </div>
  );
}

export default App;
