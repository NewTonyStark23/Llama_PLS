import { useState, useEffect } from 'react';
import { Hero } from './components/Hero';
import { Architecture } from './components/Architecture';
import { Benchmarks } from './components/Benchmarks';
import { Innovations } from './components/Innovations';
import { CodeExamples } from './components/CodeExamples';
import { GithubReadme } from './components/GithubReadme';
import { Navigation } from './components/Navigation';
import { Footer } from './components/Footer';

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
