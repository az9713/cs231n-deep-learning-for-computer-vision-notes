"""Optional browser check; requires Playwright and an available Chrome."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parent
checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome')
    page=browser.new_page()
    page.route('http://**',lambda r:r.abort())
    page.route('https://**',lambda r:r.abort())
    for width in (1280,390):
        page.set_viewport_size({'width':width,'height':900})
        page.goto((root/'numerical-lab.html').as_uri())
        page.evaluate('async()=>{await MathJax.startup.promise}')
        initial=page.locator('#temp-result').inner_text()
        for selector,value in [('#lab-temp','0.1'),('#lab-prevalence','10'),('#lab-steps','500')]:
            page.locator(selector).fill(value)
            page.locator(selector).dispatch_event('input')
        data=page.evaluate("""()=>{
          const p=visionLab.probabilities(.5), sharp=visionLab.probabilities(.1);
          const a=visionLab.alertMetrics(.01),b=visionLab.alertMetrics(.10);
          return {sum:p.reduce((a,b)=>a+b),sharper:sharp[0]>p[0],
            precision:a.precision,greaterPrevalence:b.precision>a.precision,
            endpointError:Math.abs(visionLab.integrate(2000)-2.625),
            displayedTemp:document.querySelector('#temp-value').textContent,
            displayedPrevalence:document.querySelector('#prevalence-value').textContent,
            displayedSteps:document.querySelector('#steps-value').textContent,
            overflow:document.documentElement.scrollWidth>innerWidth+2,
            mathErrors:document.querySelectorAll('mjx-merror,[data-mjx-error]').length};
        }""")
        assert abs(data['sum']-1)<1e-12 and data['sharper']
        assert abs(data['precision']-90/585)<1e-12 and data['greaterPrevalence']
        assert data['endpointError']<.002
        assert data['displayedTemp']=='0.10' and data['displayedPrevalence']=='10.0%' and data['displayedSteps']=='500'
        assert not data['overflow'] and data['mathErrors']==0
        checks.append({'viewport':width,**data})
    browser.close()
report={'status':'passed','network':'HTTP and HTTPS blocked','checks':checks}
(root/'lab-browser-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
